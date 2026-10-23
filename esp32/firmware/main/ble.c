// Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
// Bluetooth LE transport for the device API. One connection at a time. Everything the device says is a JSON line, sent as
// notifications on the TX characteristic in chunks of (MTU - 3) bytes; a client writes command lines to the RX
// characteristic (a line ends with '\n'). Nothing is sent before the link is encrypted (and, with CONFIG_TASR_BLE_PASSKEY,
// authenticated: the device shows a 6-digit PIN on its screen and the phone asks for it).
#include "ble.h"
#include "sdkconfig.h"

#ifdef CONFIG_TASR_BLE
#include <stdio.h>
#include <string.h>
#include "api.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_random.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "host/ble_hs.h"
#include "host/util/util.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "nvs_flash.h"
#include "oled.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"

extern void ble_store_config_init(void);

static const char *TAG = "ble";
static uint8_t s_own_addr;
static uint16_t s_conn = BLE_HS_CONN_HANDLE_NONE;
static uint16_t s_tx_handle;
static volatile int s_mtu = 23;
static volatile bool s_notify, s_secure;
static QueueHandle_t s_q;

typedef struct {
    char *p;
    int n;
} item_t;

// Nordic UART Service: 6E400001-B5A3-F393-E0A9-E50E24DCCA9E, RX (write) ...0002, TX (notify) ...0003
static const ble_uuid128_t svc_uuid = BLE_UUID128_INIT(0x9e, 0xca, 0xdc, 0x24, 0x0e, 0xe5, 0xa9, 0xe0, 0x93, 0xf3, 0xa3, 0xb5, 0x01, 0x00, 0x40, 0x6e);
static const ble_uuid128_t rx_uuid = BLE_UUID128_INIT(0x9e, 0xca, 0xdc, 0x24, 0x0e, 0xe5, 0xa9, 0xe0, 0x93, 0xf3, 0xa3, 0xb5, 0x02, 0x00, 0x40, 0x6e);
static const ble_uuid128_t tx_uuid = BLE_UUID128_INIT(0x9e, 0xca, 0xdc, 0x24, 0x0e, 0xe5, 0xa9, 0xe0, 0x93, 0xf3, 0xa3, 0xb5, 0x03, 0x00, 0x40, 0x6e);

#ifdef CONFIG_TASR_BLE_PASSKEY
#define RX_SEC BLE_GATT_CHR_F_WRITE_AUTHEN
#else
#define RX_SEC BLE_GATT_CHR_F_WRITE_ENC
#endif

static void advertise(void);

static int rx_access(uint16_t conn, uint16_t attr, struct ble_gatt_access_ctxt *ctxt, void *arg)
{
    (void)conn; (void)attr; (void)arg;
    if (ctxt->op != BLE_GATT_ACCESS_OP_WRITE_CHR) return BLE_ATT_ERR_UNLIKELY;
    static char line[512];
    static int n;
    uint8_t tmp[256];
    uint16_t got = 0;
    if (ble_hs_mbuf_to_flat(ctxt->om, tmp, sizeof(tmp), &got) != 0) return BLE_ATT_ERR_UNLIKELY;
    for (int i = 0; i < got; i++) {
        if (tmp[i] == '\n' || tmp[i] == '\r') {
            if (n) { line[n] = 0; n = 0; api_handle_line(line); }
        } else if (n < (int)sizeof(line) - 1) {
            line[n++] = (char)tmp[i];
        } else {
            n = 0;
        }
    }
    if (n && line[n - 1] == '}') {  // a phone app wrote one complete object without a newline
        line[n] = 0;
        n = 0;
        api_handle_line(line);
    }
    return 0;
}

static int tx_access(uint16_t conn, uint16_t attr, struct ble_gatt_access_ctxt *ctxt, void *arg)
{
    (void)conn; (void)attr; (void)ctxt; (void)arg;
    return 0;  // notify only
}

static const struct ble_gatt_svc_def s_svcs[] = {
    {.type = BLE_GATT_SVC_TYPE_PRIMARY,
     .uuid = &svc_uuid.u,
     .characteristics = (struct ble_gatt_chr_def[]){
         {.uuid = &rx_uuid.u, .access_cb = rx_access, .flags = BLE_GATT_CHR_F_WRITE | BLE_GATT_CHR_F_WRITE_NO_RSP | RX_SEC},
         {.uuid = &tx_uuid.u, .access_cb = tx_access, .val_handle = &s_tx_handle, .flags = BLE_GATT_CHR_F_NOTIFY},
         {0}}},
    {0}};

// the API's outgoing lines: queued here (never block the recognizer), sent by the tx task
static void ble_sink(const char *line, int len)
{
    if (!s_notify || !s_secure || !s_q) return;
    char *c = heap_caps_malloc(len + 1, MALLOC_CAP_SPIRAM);
    if (!c) return;
    memcpy(c, line, len);
    c[len] = '\n';
    const item_t it = {c, len + 1};
    const bool precious = !strncmp(line, "{\"type\":\"final\"", 14) || !strncmp(line, "{\"type\":\"ack\"", 12);
    if (xQueueSend(s_q, &it, precious ? pdMS_TO_TICKS(100) : 0) != pdTRUE) heap_caps_free(c);  // a full queue drops the rest
}

static void tx_task(void *arg)
{
    (void)arg;
    item_t it;
    for (;;) {
        if (xQueueReceive(s_q, &it, portMAX_DELAY) != pdTRUE) continue;
        int off = 0, tries = 0;
        while (off < it.n && s_conn != BLE_HS_CONN_HANDLE_NONE && s_notify) {
            int c = s_mtu - 3;
            if (c < 20) c = 20;
            if (c > it.n - off) c = it.n - off;
            struct os_mbuf *om = ble_hs_mbuf_from_flat(it.p + off, c);
            if (!om) { vTaskDelay(pdMS_TO_TICKS(10)); if (++tries > 50) break; continue; }
            const int rc = ble_gatts_notify_custom(s_conn, s_tx_handle, om);  // consumes om
            if (rc == 0) { off += c; tries = 0; }
            else if (rc == BLE_HS_ENOMEM || rc == BLE_HS_EBUSY || rc == BLE_HS_EAGAIN) { vTaskDelay(pdMS_TO_TICKS(5)); if (++tries > 100) break; }
            else break;
        }
        heap_caps_free(it.p);
    }
}

static int gap_event(struct ble_gap_event *ev, void *arg)
{
    (void)arg;
    switch (ev->type) {
    case BLE_GAP_EVENT_CONNECT:
        if (ev->connect.status == 0) {
            s_conn = ev->connect.conn_handle;
            s_notify = s_secure = false;
            s_mtu = 23;
            api_ble("connected");
            ble_gap_security_initiate(s_conn);  // pair / encrypt right away: nothing is sent until the link is secure
        } else {
            advertise();
        }
        break;
    case BLE_GAP_EVENT_DISCONNECT:
        s_conn = BLE_HS_CONN_HANDLE_NONE;
        s_notify = s_secure = false;
        api_ble("disconnected");
        advertise();
        break;
    case BLE_GAP_EVENT_MTU:
        s_mtu = ev->mtu.value;
        break;
    case BLE_GAP_EVENT_SUBSCRIBE:
        if (ev->subscribe.attr_handle == s_tx_handle) s_notify = ev->subscribe.cur_notify;
        break;
    case BLE_GAP_EVENT_ENC_CHANGE: {
        struct ble_gap_conn_desc d;
        s_secure = false;
        if (ev->enc_change.status == 0 && ble_gap_conn_find(ev->enc_change.conn_handle, &d) == 0) {
#ifdef CONFIG_TASR_BLE_PASSKEY
            s_secure = d.sec_state.encrypted && d.sec_state.authenticated;
#else
            s_secure = d.sec_state.encrypted;
#endif
        }
        if (s_secure) { oled_status("hold to talk"); api_ble("secure"); }
        break;
    }
    case BLE_GAP_EVENT_PASSKEY_ACTION:
        if (ev->passkey.params.action == BLE_SM_IOACT_DISP) {
            struct ble_sm_io pk = {.action = BLE_SM_IOACT_DISP, .passkey = esp_random() % 1000000};
            ble_sm_inject_io(ev->passkey.conn_handle, &pk);
            char s[24];
            snprintf(s, sizeof(s), "PIN %06u", (unsigned)pk.passkey);
            oled_status(s);  // shown on the screen only; never over the air
            ESP_LOGW(TAG, "pairing: enter PIN %06u on the phone", (unsigned)pk.passkey);
        }
        break;
    case BLE_GAP_EVENT_REPEAT_PAIRING: {  // the phone lost its bond: drop ours and pair again
        struct ble_gap_conn_desc d;
        ble_gap_conn_find(ev->repeat_pairing.conn_handle, &d);
        ble_store_util_delete_peer(&d.peer_id_addr);
        return BLE_GAP_REPEAT_PAIRING_RETRY;
    }
    case BLE_GAP_EVENT_ADV_COMPLETE:
        advertise();
        break;
    default:
        break;
    }
    return 0;
}

static void advertise(void)
{
    struct ble_hs_adv_fields f = {0};
    const char *name = ble_svc_gap_device_name();
    f.flags = BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP;
    f.name = (const uint8_t *)name;
    f.name_len = strlen(name);
    f.name_is_complete = 1;
    ble_gap_adv_set_fields(&f);
    struct ble_hs_adv_fields r = {0};
    r.uuids128 = (ble_uuid128_t *)&svc_uuid;
    r.num_uuids128 = 1;
    r.uuids128_is_complete = 1;
    ble_gap_adv_rsp_set_fields(&r);
    struct ble_gap_adv_params p = {0};
    p.conn_mode = BLE_GAP_CONN_MODE_UND;
    p.disc_mode = BLE_GAP_DISC_MODE_GEN;
    const int rc = ble_gap_adv_start(s_own_addr, NULL, BLE_HS_FOREVER, &p, gap_event, NULL);
    if (rc != 0 && rc != BLE_HS_EALREADY) ESP_LOGE(TAG, "advertising failed (%d)", rc);
}

static void on_sync(void)
{
    ble_hs_util_ensure_addr(0);
    ble_hs_id_infer_auto(0, &s_own_addr);
    advertise();
    ESP_LOGI(TAG, "advertising as \"%s\"", ble_svc_gap_device_name());
}

static void host_task(void *arg)
{
    (void)arg;
    nimble_port_run();
    nimble_port_freertos_deinit();
}

static bool ext_cmd(const char *cmd, int id)
{
    if (strcmp(cmd, "ble_unpair")) return false;
    ble_store_clear();  // forget every bonded phone
    if (s_conn != BLE_HS_CONN_HANDLE_NONE) ble_gap_terminate(s_conn, BLE_ERR_REM_USER_CONN_TERM);
    api_ack(cmd, id, true, NULL);
    return true;
}

void ble_start(void)
{
    esp_err_t e = nvs_flash_init();  // bonds and radio calibration live in NVS
    if (e == ESP_ERR_NVS_NO_FREE_PAGES || e == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        nvs_flash_erase();
        e = nvs_flash_init();
    }
    if (e != ESP_OK) { ESP_LOGE(TAG, "nvs init failed"); return; }
    s_q = xQueueCreate(24, sizeof(item_t));
    if (nimble_port_init() != ESP_OK) { ESP_LOGE(TAG, "nimble init failed"); return; }
    ble_hs_cfg.sync_cb = on_sync;
    ble_hs_cfg.store_status_cb = ble_store_util_status_rr;
    ble_hs_cfg.sm_bonding = 1;
    ble_hs_cfg.sm_sc = 1;
    ble_hs_cfg.sm_our_key_dist = BLE_SM_PAIR_KEY_DIST_ENC | BLE_SM_PAIR_KEY_DIST_ID;
    ble_hs_cfg.sm_their_key_dist = BLE_SM_PAIR_KEY_DIST_ENC | BLE_SM_PAIR_KEY_DIST_ID;
#ifdef CONFIG_TASR_BLE_PASSKEY
    ble_hs_cfg.sm_io_cap = BLE_SM_IO_CAP_DISP_ONLY;  // the screen shows the PIN: protects against a man in the middle
    ble_hs_cfg.sm_mitm = 1;
#else
    ble_hs_cfg.sm_io_cap = BLE_SM_IO_CAP_NO_IO;  // "just works": encrypted, but anyone nearby can pair (development)
    ble_hs_cfg.sm_mitm = 0;
#endif
    ble_att_set_preferred_mtu(247);
    ble_svc_gap_init();
    ble_svc_gatt_init();
    ble_gatts_count_cfg(s_svcs);
    ble_gatts_add_svcs(s_svcs);
    ble_svc_gap_device_name_set(CONFIG_TASR_BLE_NAME);
    ble_store_config_init();
    api_add_sink(ble_sink);
    api_set_ext_handler(ext_cmd);
    xTaskCreatePinnedToCore(tx_task, "ble_tx", 3072, NULL, 4, NULL, TASR_AUX_CORE);
    nimble_port_freertos_init(host_task);
}
#else
void ble_start(void) {}
#endif
