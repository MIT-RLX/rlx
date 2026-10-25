/* RLX — versatile ML compiler + runtime.
 * Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
 * SPDX-License-Identifier: MIT OR Apache-2.0 */
#include <stdio.h>

#include "port.h"

void port_write(const char* s) { fputs(s, stdout); }
