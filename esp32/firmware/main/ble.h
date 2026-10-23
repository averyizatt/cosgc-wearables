// Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
// Bluetooth LE transport for the device API (docs/API.md): a GATT service in the Nordic UART layout (so generic apps such
// as nRF Connect work with it) that carries the same JSON lines as the USB port. No-ops unless CONFIG_TASR_BLE is set.
#pragma once

void ble_start(void);  // initialize NimBLE and start advertising (call once, early: it needs internal RAM)
