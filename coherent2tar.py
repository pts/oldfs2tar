#! /bin/sh
# by pts@fazekas.hu at Mon Feb  6 16:57:24 CET 2023

""":" # coherent2tar.py: copy files from a Coherent filesystem

type python2.7 >/dev/null 2>&1 && exec python2.7 -- "$0" ${1+"$@"}
type python2.6 >/dev/null 2>&1 && exec python2.6 -- "$0" ${1+"$@"}
type python2.5 >/dev/null 2>&1 && exec python2.5 -- "$0" ${1+"$@"}
type python2.4 >/dev/null 2>&1 && exec python2.4 -- "$0" ${1+"$@"}
exec python -- ${1+"$@"}; exit 1

This script need Python 2.4, 2.5, 2.6 or 2.7. Python 3.x won't work.

Typical usage: coherent2tar.py coh-2.3.43-boot.dsk coh-2.3.43-boot.tar

This program can read filesystems created by Coherent 2.x and 3.x. Newer
versions of Coherent were not tested.

This program can create a .tar file containing everything in the Coherent
filesystem.

This program can be run as a regular user if it has permission to read the
filesystem disk image.

The Linux kernel can also mount Coherent 2.x and 3.x filesystems, but (1) it
doesn't show device node major and minor numbers correctly; (2) it can't
mount the Coherent 2.3.43 setup disk images (*.dsk), see
reorder_coherent_dsk.py for converting them.
"""

import array
import struct
import sys
import tarfile


class Struct(object):
  """A structure serializable to a byte string of fixed size."""

  SIZE = FMT = FIELDS = None  # Initialzed in new_struct_class(...).

  def __init__(self, data=''):
    if data == '':
      data = '\0' * self.SIZE
    self.loads(data)

  def loads(self, data):
    """Parse from byte string."""
    if len(data) != self.SIZE:
      raise ValueError('Bad struct %s size: expected=%d got=%d' % (self.__class__.__name__, self.SIZE, len(data)))
    fmt0 = self.FMT[0]  # Byte order: '<' or '>'.
    ofs = 0
    for fmt, fname, doc in self.FIELDS:
      if not fmt:
        raise ValueError('Empty field format.')
      fmt1 = fmt0 + fmt.replace('M', 'L').replace('K', 'B')
      size = struct.calcsize(fmt1)
      value = struct.unpack(fmt1, buffer(data, ofs, size))
      ofs += size
      fmtm1 = fmt[-1]
      if fmtm1 == 'M':  # Word-swapped uint32_t.
        value = [(v >> 16) & 0xffff | (v & 0xffff) << 16 for v in value]
      elif fmtm1 == 'K':  # uint24_t in Coherent byte order.
        value = [value[i] << 16 | value[i + 1] | value[i + 2] << 8 for i in xrange(0, len(value), 3)]
      if fmtm1 == 's':
        value = value[0].rstrip('\0')
      elif len(value) == 1 and fmt[0] not in '123456789':
        value = int(value[0])
      else:
        value = [int(v) for v in value]
      setattr(self, fname, value)

  def __repr__(self):
    output = []
    for fmt, fname, doc in self.FIELDS:
      if output:
        output.append(', ')
      output.extend((fname, '=', repr(getattr(self, fname))))
    return '%s[%s]' % (self.__class__.__name__, ''.join(output))

  def __eq__(self, other):  # Also defines `__ne__'.
    return self.dumps() == other.dumps()

  def dumps(self):
    """Serialize and return byte string."""
    fmt0 = self.FMT[0]  # Byte order: '<' or '>'.
    i, output = 0, []
    for fmt, fname, doc in self.FIELDS:
      value = getattr(self, fname)
      fmtm1 = fmt[-1]
      if fmtm1 == 'M':  # Word-swapped uint32_t.
        if not isinstance(value, (list, tuple)):
          value = (value,)
        value = [(v >> 16) & 0xffff | (v & 0xffff) << 16 for v in value]
      elif fmtm1 == 'K':  # uint24_t in Coherent byte order.
        vs = value
        if not isinstance(value, (list, tuple)):
          vs = (vs,)
        value = []
        for v in vs:
          if not isinstance(v, (int, long)):
            raise TypeError('Bad type of vs for K field: %r', (v,))
          if not (0 <= v < (1 << 24)):
            raise ValueError('Value out of range for K field: %d' % v)
          value.extend(((v >> 16) & 0xff, v & 0xff, (v >> 8) & 0xff))
        vs = ()  # Save memory.
      if isinstance(value, (list, tuple)):
        output.extend(value)
      else:
        output.append(value)
    return struct.pack(self.FMT, *output)


def new_struct_class(name, doc, fields):
  for fmt, fname, doc in fields:
    if not fmt:
      raise ValueError('Empty field format in field: %s.%s' % (name, fname))
    elif fmt.endswith('K'):
      try:
        if int(fmt[:-1]) % 3:
          raise ValueError
      except ValueError:
        raise ValueError('K field must have a count divisible by 3, got %s: %s.%s' % (fmt, name, fname))
  struct_class = type(name, (Struct,), {'__doc__': str(doc)})
  # TODO(pts): Initialize struct_class.__slots__?
  struct_class.FIELDS = fields
  struct_class.FMT = fmt = '<' + ''.join(fmt for fmt, fname, doc in fields).replace('M', 'L').replace('K', 'B')
  struct_class.SIZE = struct.calcsize(fmt)
  return struct_class


# --- Coherent filesystem structures.

NICINOD = 100  # Number of free in core inodes
BSIZE   = 512  # Block size
BSHIFT  = 9    # BSIZE == (1 << BSHIFT).
INOPB   = 8    # Number of inodes per block
BOOTBI  = 0    # Boot block index. The boot block is not extracted.
SUPERI  = 1    # Super block index
INODEI  = 2    # Inode block index
BADFIN  = 1    # Bad block inode number
ROOTIN  = 2    # Root inode number
NICFREE = 64   # Number of blocks in a free block list
MAXINTN = 255  # maptab must be int * if > 255

# Inode.di_mode bit values.
S_IFMT  = 0170000  # Type
S_IFDIR = 0040000  # Directory
S_IFCHR = 0020000  # Character special
S_IFBLK = 0060000  # Block special
S_IFREG = 0100000  # Regular
S_IFMPC = 0030000  # Multiplexed character special
S_IFMPB = 0070000  # Multiplexed block special
S_IFPIP = 0010000  # Pipe

# 'M' is 'L' (uint32), but stored in word-swapped canonical order (_canl).
# daddr_t is 'L' uint32_t.
# time_t is 'l' int32_t.
# ino_t is 'H' uint16_t.
# fsize_t is 'L' uint32.
# dev_t is 'H' uint16_t.


Superblock = new_struct_class('Superblock', """Coherent filesystem superblock (struct filsys).""", (
    ('H',    's_isize',  'First block not in inode list, in 512-byte blocks.'),
    ('M',    's_fsize',  'daddr_t: Size of entire volume, in 512-byte blocks.'),
    ('H',    's_nfree',  'Number of addresses in s_free'),
    ('64M',  's_free',   'daddr_t: Free block list [NICFREE = 64]'),
    ('H',    's_ninode', 'Number of inodes in s_inode'),
    ('100H', 's_inode',  'ino_t: Free inode list [NICINOD = 100]'),
    ('B',    's_flock',  'Not used s_flock'),
    ('B',    's_ilock',  'Not used s_ilock'),
    ('B',    's_fmod',   'Super block modified flag'),
    ('B',    's_ronly',  'Mounted read only flag'),
    ('M',    's_time',   'time_t: Last super block update'),
    ('M',    's_tfree',  'daddr_t: Total free blocks'),
    ('H',    's_tinode', 'Total free inodes'),
    ('H',    's_m',      'Interleave factor m'),
    ('H',    's_n',      'Interleave factor n'),
    ('6s',   's_fname',  'File system name [6]'),
    ('6s',   's_fpack',  'File system pack name [6]'),
    ('M',    's_unique', 'Unique number'),
    ('12s',  's_pad',    'Padding to 512 bytes [12]'),
))
assert Superblock.SIZE == 0x200


Inode = new_struct_class('Inode', """Coherent filesystem inode (struct dinode).""", (
    ('H',   'di_mode',  'Mode'),
    ('H',   'di_nlink', 'Link count'),
    ('H',   'di_uid',   'User id of owner'),
    ('H',   'di_gid',   'Group id of owner'),
    ('M',   'di_size',  'fsize_t: Size of file in bytes'),
    # Overlaps with di_addd.
    #('H',  'di_rdev',  'Device'),
    ('30K', 'di_addd',  'Disk block addresses: 10*3 bytes direct [30]'),
    ('9K',  'di_addi',  'Disk block addresses: 3*3 bytes indirect [3]'),
    ('B',   'di_addp',  'Disk block addresses: padding to 40 bytes'),
    ('M',   'di_atime', 'Last access time'),
    ('M',   'di_mtime', 'Last modify time'),
    ('M',   'di_ctime', 'Last creation time'),
))
assert Inode.SIZE == 0x40


Dentry = new_struct_class('Dentry', """Coherent filesystem directory entry (struct direct). """, (
    ('H',   'd_ino',  'ino_t: Inode number'),
    ('14s', 'd_name', 'Name'),
))
assert Dentry.SIZE == 0x10


class Interleave(object):  # !! TODO(pts): Is this needed? (We don't have an image to test.)
  """Implements block interleave table."""

  __slots__ = ('maptab', 'mapbot', 'maptop', 'mapn')

  def __init__(self, sb):
    if not isinstance(sb, Superblock):
      raise TypeError('Superblock expected.')
    if not (1 <= sb.s_n <= MAXINTN):
      raise ValueError('Bad sb.s_n range.')
    if not (1 <= sb.s_m <= sb.s_n):
      raise ValueError('Bad sb.s_m range.')
    if sb.s_n % sb.s_m != 0:
      raise ValueError('Bad sb.s_m modulo.')
    # This is a no-op of sb.s_m == sb.s_n == 1.
    self.mapn = sb.s_n
    self.maptab = maptab = array.array('B', (0,)) * sb.s_n
    self.mapbot = ((sb.s_isize + sb.s_n - 1) // sb.s_n) * sb.s_n
    self.maptop = (sb.s_fsize // sb.s_n) * sb.s_n
    ints = sb.s_n / sb.s_m
    for i in xrange(0, sb.s_n):
      maptab[i] = (i // ints) + (i % ints) * sb.s_m

  def bmap(self, b):
    if self.mapbot <= b < self.maptop:
      i = b % self.mapn
      return b - i + self.maptab[i]
    return b


# ---


class IterFile(object):
  """Converts an iterator yielding byte strings to a file-like object."""

  __slots__ = ('it',)

  def __init__(self, it):
    self.it = iter(it)

  def read(self, size):
    output, it = [], self.it
    while size > 0:
      output.append(it.next())
      size -= len(output[-1])
      if size < 0:
        raise RuntimeError('Iterator yielded too much.')
    return ''.join(output)


class TellFile(object):
  """A writable file which can tell its own position."""

  __slots__ = ('f', 'ofs')

  def __init__(self, f, ofs=0):
    self.f = f
    self.ofs = ofs

  def write(self, data):
    self.f.write(data)
    self.ofs += len(data)

  def tell(self):
    return self.ofs

  def close(self):
    self.f.close()


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


def sector_identity(sector_idx):
  return sector_idx


MbrEntry = new_struct_class('MbrEntry', """MBR partition entry.""", (
    ('B', 'active',       '0x00 is non-bootable, 0x80 is bootable.'),
    ('B', 'first_head',   'First sector head.'),
    ('B', 'first_sector', 'First sector sector. Also contains bits of track.'),
    ('B', 'first_track',  'First sector track.'),
    ('B', 'type',         'Partition type.'),
    ('B', 'last_head',    'Last sector head.'),
    ('B', 'last_sector' , 'Last sector sector. Also contains bits of track.'),
    ('B', 'last_track',   'Last sector track.'),
    ('L', 'first_lba',    'First sector, 0-based, LBA.'),
    ('L', 'sector_count', 'Number of 512-byte sectors.'),
))


def extract(fs_filename, base_block_idx, block_size_limit, tar_filename, progressf, do_transform_sector_idx, partition):
  f = open(fs_filename, 'rb')
  try:
    def read_block(block_idx):
      if block_idx < 0:
        raise ValueError('Block index must not be negative.')
      if sb is not None and block_idx > sb.s_fsize:
        raise ValueError('Block index beyond filesystem.')
      if block_size_limit is not None and block_idx > block_size_limit:
        raise ValueError('Block index must be smaller than the size limit.')
      block_idx2 = sector_transform(block_idx) + (base_block_idx or 0)
      f.seek(block_idx2 << BSHIFT)
      data = f.read(BSIZE)
      if len(data) != BSIZE:
        raise ValueError('EOF before end of block %d: %s', block_idx, f.name)
      return data

    def read_data_block(block_idx, is_sparse_ok):
      if block_idx < sb.s_isize:
        if block_idx == 0:
          if is_sparse_ok:  # Block from a sparse file.
            return '\0' * 0x200
          else:
            raise ValueError('Block index must not be zero.')
        elif block_idx < 0:
          raise ValueError('Block index must be positive, got: %d' % block_idx)
        else:
          raise ValueError('Expected data block, found inode block: %d < %d' % (block_idx, sb.s_isize))
      return read_block(block_idx)

    def read_inode(inode_idx):
      if inode_idx < ROOTIN:
        raise ValueError('Inode number too small: %d' % inode_idx)
      block_idx = 2 + ((inode_idx - 1) >> 3)
      assert block_idx > 0
      if block_idx >= sb.s_isize:
        raise ValueError('Expected data block, found inode block.')
      data = read_block(block_idx)  # TODO(pts): Read only 0x40 bytes rather than 0x200 bytes.
      i = ((inode_idx - 1) & 7) << 6
      return data[i : i + 0x40]

    def yield_read_indirect(block_idx, count):
      if count > 0x80:
        raise ValueError('Indirect read longer than a single block.')
      # TODO(pts): Unpack less than 128 bytes.
      return ((v >> 16) & 0xffff | (v & 0xffff) << 16 for v in
              struct.unpack('<128L', read_data_block(block_idx, False))[:count])

    def yield_read_file(size, di_addd, di_addi, is_sparse_ok):
      """Yields the file one block at a time."""
      block_count = (size + 0x1ff) >> 9
      if block_count != 0:
        if block_count > 0x20408a:  # 1082201088 bytes, a bit more than 1 GiB.
          raise ValueError('File too long.')
        if block_count > 0xa:  # Add blocks from single indirect block.
          block_count2 = min(0x80, block_count - 0xa)
          di_addd = list(di_addd)
          di_addd.extend(yield_read_indirect(di_addi[0], block_count2))
        if block_count > 0x8a:  # Add blocks from double indirect block.
          block_count2 = block_count - 0x8a
          block_count3 = (block_count2 + 0x7f) >> 7
          for block_idx2 in yield_read_indirect(di_addi[1], block_count3):
            count2 = min(0x80, block_count2)
            di_addd.extend(yield_read_indirect(block_idx2, count2))
            block_count2 -= count2
        if block_count > 0x408a:  # Add blocks from triple indirect block.
          # TODO(pts): Test this.
          block_count2 = block_count - 0x408a
          block_count3 = (block_count2 + 0x7f) >> 7
          block_count4 = (block_count3 + 0x7f) >> 7
          for block_idx3 in yield_read_indirect(di_addi[3], block_count4):
            count3 = min(0x80, block_count3)
            block_count3 -= count3
            for block_idx2 in yield_read_indirect(block_idx3, count3):
              count2 = min(0x80, block_count2)
              di_addd.extend(yield_read_indirect(block_idx2, count2))
              block_count2 -= count2
        block_i = 0
        while 1:
          block_idx = di_addd[block_i]
          data = read_data_block(di_addd[block_i], is_sparse_ok)
          if block_i == block_count - 1:
            yield data[:(size & 0x1ff) or 0x200]
            break
          yield data
          block_i += 1

    def process_dir(prefix, inode_idx, inode_idx_stack, ino=None, parent_inode_idx=None):
      if inode_idx in inode_idx_stack:
        raise ValueError('Infinite recursion.')
      do_add_rootdir = ino is None
      if ino is None:
        ino = Inode(read_inode(inode_idx))
      if (ino.di_mode & S_IFMT) != S_IFDIR:
        raise ValueError('Directory inode expected.')
      if ino.di_size & 0xf:
        raise ValueError('Directory size must be a multiple of 16: %d' % ino.di_size)
      if do_add_rootdir and tf is not None:
        ti = tarfile.TarInfo()
        ti.name = './'
        ti.mtime = ino.di_mtime
        ti.mode = ino.di_mode & 07777
        ti.uid = ino.di_uid
        ti.gid = ino.di_gid
        ti.size = 0
        ti.type = tarfile.DIRTYPE
        tf.addfile(ti)
      for data in yield_read_file(ino.di_size, ino.di_addd, ino.di_addi, False):
        for ofs in xrange(0, len(data), 0x10):
          de = Dentry(buffer(data, ofs, 0x10))
          if '\0' in de.d_name or '/' in de.d_name:
            raise ValueError('Bad filename in directory entry: %r' % de.d_name)
          if de.d_name == '.':
            if de.d_ino != inode_idx:
              raise ValueError('Bad inode number for current.')
            continue
          elif de.d_name == '..':
            if de.d_ino != parent_inode_idx:
              raise ValueError('Bad inode number for parent.')
            continue
          if de.d_ino == 0:
            continue
          ino2 = Inode(read_inode(de.d_ino))
          ifmt2 = ino2.di_mode & S_IFMT
          if tf is not None:
            ti = tarfile.TarInfo()
            ti.name = prefix.lstrip('/') + de.d_name
            ti.mtime = ino2.di_mtime
            ti.mode = ino2.di_mode & 07777
            ti.uid = ino2.di_uid
            ti.gid = ino2.di_gid
            ti.size = ino2.di_size
          if ifmt2 == S_IFDIR:
            if tf is not None:
              ti.type = tarfile.DIRTYPE
              ti.name += '/'
              ti.size = 0
              tf.addfile(ti)
            inode_idx_stack.append(inode_idx)
            try:
              process_dir(''.join((prefix, de.d_name, '/')), de.d_ino, inode_idx_stack, ino2, inode_idx)
            finally:
              inode_idx_stack.pop()
          elif ifmt2 == S_IFREG:  # TODO(pts): Add hard links for non-regular files?
            progressf.write('%s%s %d\n' % (prefix, de.d_name, ino2.di_size))
            if tf is not None:
              if de.d_ino in inode_map:
                ti.type = tarfile.LNKTYPE
                ti.linkname = inode_map[de.d_ino]
                tf.addfile(ti)
              else:
                ti.type = tarfile.REGTYPE
                inode_map[de.d_ino] = ti.name
                tf.addfile(ti, IterFile(yield_read_file(ino2.di_size, ino2.di_addd, ino2.di_addi, True)))
          elif ifmt2 == S_IFCHR:
            major, minor = ino2.di_addd[0] & 0xff, ino2.di_addd[0] >> 16
            progressf.write('%s%s char %d:%d\n' % (prefix, de.d_name, major, minor))
            if ino2.di_size:
              raise ValueError('Character device must have 0 size.')
            if tf is not None:
              ti.type = tarfile.CHRTYPE
              ti.devmajor, ti.devminor = major, minor
              tf.addfile(ti)
          elif ifmt2 == S_IFBLK:
            major, minor = ino2.di_addd[0] & 0xff, ino2.di_addd[0] >> 16
            progressf.write('%s%s block %d:%d\n' % (prefix, de.d_name, major, minor))
            if ino2.di_size:
              raise ValueError('Block device must have 0 size.')
            if tf is not None:
              ti.type = tarfile.BLKTYPE
              ti.devmajor, ti.devminor = major, minor
              tf.addfile(ti)
          elif ifmt2 == S_IFPIP:
            progressf.write('%s%s pipe\n' % (prefix, de.d_name))
            if ino2.di_size:
              raise ValueError('Pipe must have 0 size.')
            if tf is not None:
              ti.type = tarfile.FIFOTYPE
              tf.addfile(ti)
          elif ino2.di_mode & 01000:  # Symlink? Not found. TODO(pts): Is it supported by Coherent?
            progressf.write('%s%s symlink 0x%x %d ?\n' % (prefix, de.d_name, ifmt2, ino2.di_size))
            raise ValueError('Symlink not supported.')
          else:
            progressf.write('%s%s 0x%x %d ?\n' % (prefix, de.d_name, ifmt2, ino2.di_size))
            raise ValueError('Unknown file mode 0%o' % ino2.di_mode)
          progressf.flush()

    sector_transform = sector_identity  # Good enough for the superblock.
    sb = None
    if base_block_idx is None and block_size_limit is None and do_transform_sector_idx in (None, False) and partition is None:  # Detect MBR partition table, use first active partition.
      mbr_block = read_block(0)
      if mbr_block.endswith('\x55\xaa'):
        partitions = [MbrEntry(buffer(mbr_block, i, 0x10)) for i in xrange(0x1be, 0x1fe, 0x10)]
        if not [1 for p in partitions if p.active not in (0, 0x80) and (p.first_lba == 0) == (p.sector_count == 0)]:
          partitions = [p for p in partitions if p.first_lba != 0 and p.active]
          if partitions:
            # No overlap.
            if not [1 for pi in xrange(len(partitions)) for pj in xrange(pi + 1, len(partitions))
                   if max(partitions[pi].first_lba, partitions[pj].first_lba) < min(partitions[pi].first_lba + partitions[pi].sector_count, partitions[pj].first_lba + partitions[pj].sector_count)]:
              # Now check that partitions don't extend beyond the end of the disk.
              f.seek(0, 2)  # Seek to EOF.
              block_count = f.tell() >> 9
              if not [1 for p in partitions if p.first_lba + p.sector_count > block_count]:
                # This looks like an MBR.
                base_block_idx, block_size_limit = partitions[0].first_lba, partitions[0].sector_count
        partitions = None  # Save memory.
      mbr_block = None  # Save memory.
    elif partition is not None:
      if partition and not (base_block_idx is None and block_size_limit is None and do_transform_sector_idx in (None, False)):
        raise ValueError('Asking for a partition conflicts with other settings.')
      if partition:
        if not (1 <= partition <= 4):
          raise ValueError('Bad partition index: %d' % partition)
        mbr_block = read_block(0)
        if mbr_block.endswith('\x55\xaa'):
          partitions = [MbrEntry(buffer(mbr_block, i, 0x10)) for i in xrange(0x1be, 0x1fe, 0x10)]
          if not [1 for p in partitions if p.active not in (0, 0x80) and (p.first_lba == 0) == (p.sector_count == 0)]:
            p = partitions[partition - 1]
            if p.first_lba == 0:
              raise ValueError('Missing partition: %d' % partition)
            base_block_idx, block_size_limit = p.first_lba, p.sector_count
          partitions = None  # Save memory.
        mbr_block = None  # Save memory.
    if base_block_idx is None:
      base_block_idx = 0

    sb = Superblock(read_block(SUPERI))
    # il = Interleave(sb)  # !! TODO(pts): Is this needed? It's a no-op for all our test files (with sb.s_m == sb.s_n == 1).
    if do_transform_sector_idx is None:
      do_transform_sector_idx = (base_block_idx == 0 and sb.s_fsize == 720)  # coh-2.3.43-boot.dsk, coh-2.3.43-d1.dsk etc. 368640 bytes.
    sector_transform = (sector_identity, sector_dsk_to_lba)[do_transform_sector_idx]
    if tar_filename is None:
      tf = None
    else:
      inode_map = {}
      sys.stderr.write('info: creating tar file: %s\n' % tar_filename)
      kwargs = {}
      if getattr(tarfile, 'USTAR_FORMAT', None) is not None:  # Missing but default in Python 2.4.
        kwargs['format'] = tarfile.USTAR_FORMAT  # Standardized in 1988. Doesn't support ctime and atime.
      if tar_filename == '-':
        tf = tarfile.TarFile(None, 'w', fileobj=TellFile(sys.stdout), **kwargs)
      else:
        tf = tarfile.TarFile(tar_filename, 'w', **kwargs)
    try:
      process_dir('/', ROOTIN, [], None, ROOTIN)
    finally:
      if tf is not None:
        tf.close()
  finally:
    f.close()


def main(argv):
  if len(argv) < 2 or argv[1] == '--help':
    sys.stderr.write(
        'coherent2tar.py: copy files from a Coherent filesystem\n'
        'This is free software, GNU GPL >=2.0. '
        'There is NO WARRANTY. Use at your risk.\n'
        'Usage: %s [<flag> ...] <filesystem-image> [<output-tar-filename>]\n'
        'Flags:\n'
        '--offset=<byte-offset>\n'
        '--base-block-idx=<block-idx>: Of 512 bytes.\n'
        '--block-size-limit=<block-count>: Of 512 bytes.\n'
        '--[no-]transform-sector-idx: Support Coherent 2.3.43 .dsk format.\n'
        '--partition=<idx>: Primary partition 1, 2, 3 or 4.\n'
        % (argv[0],))
    sys.exit(len(argv) < 2)

  i = 1
  base_block_idx = None
  block_size_limit = None
  do_transform_sector_idx = None
  partition = None
  while i < len(argv):
    arg = argv[i]
    i += 1
    if arg == '--':
      break
    elif not arg.startswith('-') or arg == '-':
      i -= 1
      break
    elif arg.startswith('--offset='):
      ofs = int(arg[arg.find('=') + 1:])
      if ofs & 0x1ff:
        sys.exit('fatal: offset must be a multiple of 512: %d' % ofs)
      base_block_idx = ofs >> 9
    elif arg.startswith('--base-block-idx='):  # Useful for extracting a partition.
      base_block_idx = int(arg[arg.find('=') + 1:])
    elif arg.startswith('--block-size-limit='):  # Useful for extracting a partition.
      base_block_idx = int(arg[arg.find('=') + 1:])
    elif arg.startswith('--partition='):
      partition = int(arg[arg.find('=') + 1:])
      if not (0 <= partition <= 4):
        sys.exit('fatal: bad partition index: %d' % partition)
    elif arg.startswith('--transform-sector-idx'):
      do_transfor_sector_idx = True
    elif arg.startswith('--no-transform-sector-idx'):
      do_transfor_sector_idx = False
    else:
      sys.exit('fatal: unknown flag: %s' % arg)
  if base_block_idx is not None and base_block_idx < 0:
    sys.exit('fatal: base block must be nonnegative: %d' % base_block_idx)

  if i >= len(argv):
    sys.exit('fatal: missing <filesystem-image> in command line')
  fs_filename = argv[i]
  i += 1
  if i >= len(argv):
    tar_filename = None
  else:
    tar_filename = argv[i]
    i += 1
  if i < len(argv):
    sys.exit('fatal: too many command-line arguments')
  progressf = (sys.stdout, sys.stderr)[tar_filename == '-']

  extract(fs_filename, base_block_idx, block_size_limit, tar_filename, progressf, do_transform_sector_idx, partition)


if __name__ == '__main__':
  sys.exit(main(sys.argv))
