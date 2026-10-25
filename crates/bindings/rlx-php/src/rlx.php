<?php
// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// rlx for PHP: require this file (composer's autoload does). Needs the FFI
// extension with `ffi.enable=1` (or `preload`) on the CLI.

declare(strict_types=1);

require_once __DIR__ . '/Support.php';
require_once __DIR__ . '/Manual.php';
require_once __DIR__ . '/Generated.php';
