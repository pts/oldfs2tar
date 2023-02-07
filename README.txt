oldfs2tar: extract filesystem images to .tar files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
oldfs2tar is a set of command-line tools written in Python 2 to extract
filesystem images to .tar files.

The tools need Python 2.4, 2.5, 2.6 or 2.7. They don't work in Python 3.

Included tools:

* coherent2tar.py: extract from Coherent filesystem. Tested with Coherent
  2.3.43 and 3.2.1 filesystems.

Example invocation:

  $ wget https://www.autometer.de/unix4fun/coherent/ftp/distrib/Coherent-2.3.43/coh-2.3.43-boot.dsk
  $ ./coherent2tar.py coh-2.3.43-boot.dsk coh-2.3.43-boot.tar

__END__
