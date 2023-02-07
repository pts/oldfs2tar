#! /usr/bin/python
#
# reorder_coherent_dsk.py: reorder Coherent 2.3.43 .dsk disk images so that Linux can mount them
# by pts@fazekas.hu at Tue Feb  7 03:30:21 CET 2023
#
# This script need Python 2.4, 2.5, 2.6 or 2.7. Python 3.x won't work.
#
# Example invocation:
#
#   $ wget https://www.autometer.de/unix4fun/coherent/ftp/distrib/Coherent-2.3.43/coh-2.3.43-boot.dsk
#   $ ./reorder_coherent_dsk.py coh-2.3.43-boot.dsk
#   $ mkdir mp
#   $ sudo mount -t sysv -o loop,ro coh-2.3.43-boot.dsk.img mp
#

import sys

def sector_dsk_to_lba(sector_idx):
  # 368640 bytes == 360 KiB == 2 [heads_per_cylinder] * 40 [tracks] * 9 [sectors_per_track] * 1/2 KiB == 720 sectors
  # We return the sector index in the LBA format ( https://en.wikipedia.org/wiki/Logical_block_addressing#CHS_conversion ):
  #    S0 == 0-based sector number; C0 == 0-based cylinder number; H0 == 0-based head number.
  #    LBA == (C0 * heads_per_cylinder + H0) * sectors_per_track + S0
  tracks = 40
  sectors_per_track = 9
  sectors_per_cylinder = tracks << 1  # heads = 2.
  adr = (sector_idx // sectors_per_track) % sectors_per_cylinder
  if adr >= tracks:
    adr -= sectors_per_cylinder - 1
  return sector_idx + adr * sectors_per_track


def main(argv):
  for dsk_filename in sys.argv[1:]:
    inf = open(dsk_filename, 'rb')
    outf = open(dsk_filename + '.img', 'wb')
    outf.truncate(0)  # Remove old contents.
    for lba in xrange(720):
      ofs = sector_dsk_to_lba(lba)
      inf.seek(ofs << 9)
      data = inf.read(0x200)
      if len(data) != 0x200:
        raise ValueError
      outf.write(data)


if __name__ == '__main__':
  sys.exit(main(sys.argv))
