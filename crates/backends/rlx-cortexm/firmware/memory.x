/* nRF52840 with the Nordic open USB bootloader present.
 *
 * 0x00000000 .. 0x00001000   MBR
 * 0x00001000 .. 0x000F4000   Application (this firmware)   ─── 972 KB
 * 0x000F4000 .. 0x00100000   Open USB bootloader + settings page
 *
 * If you flash via SWD (no bootloader), set FLASH ORIGIN = 0x00000000
 * LENGTH = 1024K instead.
 */
MEMORY
{
  FLASH : ORIGIN = 0x00001000, LENGTH = 972K
  RAM   : ORIGIN = 0x20000000, LENGTH = 256K
}
