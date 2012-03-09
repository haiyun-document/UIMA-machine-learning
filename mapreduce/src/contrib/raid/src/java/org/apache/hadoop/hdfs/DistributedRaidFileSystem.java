/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.hadoop.hdfs;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.DataInput;
import java.io.PrintStream;
import java.net.URI;
import java.net.URISyntaxException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.permission.FsPermission;
import org.apache.hadoop.fs.ChecksumException;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FilterFileSystem;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSInputStream;
import org.apache.hadoop.hdfs.BlockMissingException;
import org.apache.hadoop.hdfs.DistributedFileSystem;
import org.apache.hadoop.util.Progressable;
import org.apache.hadoop.util.ReflectionUtils;
import org.apache.hadoop.raid.Decoder;
import org.apache.hadoop.raid.RaidNode;
import org.apache.hadoop.raid.ReedSolomonDecoder;
import org.apache.hadoop.raid.XORDecoder;
import org.apache.hadoop.raid.protocol.PolicyInfo.ErasureCodeType;

/**
 * This is an implementation of the Hadoop  RAID Filesystem. This FileSystem 
 * wraps an instance of the DistributedFileSystem.
 * If a file is corrupted, this FileSystem uses the parity blocks to 
 * regenerate the bad block.
 */

public class DistributedRaidFileSystem extends FilterFileSystem {

  // these are alternate locations that can be used for read-only access
  DecodeInfo[] alternates;
  Configuration conf;
  int stripeLength;

  DistributedRaidFileSystem() throws IOException {
  }

  DistributedRaidFileSystem(FileSystem fs) throws IOException {
    super(fs);
    alternates = null;
    stripeLength = 0;
  }

  // Information required for decoding a source file
  static private class DecodeInfo {
    final Path destPath;
    final ErasureCodeType type;
    final Configuration conf;
    final int stripeLength;
    private DecodeInfo(Configuration conf, ErasureCodeType type, Path destPath) {
      this.conf = conf;
      this.type = type;
      this.destPath = destPath;
      this.stripeLength = RaidNode.getStripeLength(conf);
    }

    Decoder createDecoder() {
      if (this.type == ErasureCodeType.XOR) {
        return new XORDecoder(conf, stripeLength);
      } else if (this.type == ErasureCodeType.RS) {
        return new ReedSolomonDecoder(conf, stripeLength,
                              RaidNode.rsParityLength(conf));
      }
      return null;
    }
  }

  /* Initialize a Raid FileSystem
   */
  public void initialize(URI name, Configuration conf) throws IOException {
    this.conf = conf;

    Class<?> clazz = conf.getClass("fs.raid.underlyingfs.impl",
        DistributedFileSystem.class);
    if (clazz == null) {
      throw new IOException("No FileSystem for fs.raid.underlyingfs.impl.");
    }
    
    this.fs = (FileSystem)ReflectionUtils.newInstance(clazz, null); 
    super.initialize(name, conf);
    
    // find stripe length configured
    stripeLength = RaidNode.getStripeLength(conf);
    if (stripeLength == 0) {
      LOG.info("dfs.raid.stripeLength is incorrectly defined to be " + 
               stripeLength + " Ignoring...");
      return;
    }

    // Put XOR and RS in alternates
    alternates= new DecodeInfo[2];
    Path xorPath = RaidNode.xorDestinationPath(conf, fs);
    alternates[0] = new DecodeInfo(conf, ErasureCodeType.XOR, xorPath);
    Path rsPath = RaidNode.rsDestinationPath(conf, fs);
    alternates[1] = new DecodeInfo(conf, ErasureCodeType.RS, rsPath);
  }

  /*
   * Returns the underlying filesystem
   */
  public FileSystem getFileSystem() throws IOException {
    return fs;
  }

  @Override
  public FSDataInputStream open(Path f, int bufferSize) throws IOException {
    ExtFSDataInputStream fd = new ExtFSDataInputStream(conf, this, alternates, f,
                                                       stripeLength, bufferSize);
    return fd;
  }

  public void close() throws IOException {
    if (fs != null) {
      try {
        fs.close();
      } catch(IOException ie) {
        //this might already be closed, ignore
      }
    }
    super.close();
  }

  /**
   * Layered filesystem input stream. This input stream tries reading
   * from alternate locations if it encoumters read errors in the primary location.
   */
  private static class ExtFSDataInputStream extends FSDataInputStream {
    /**
     * Create an input stream that wraps all the reads/positions/seeking.
     */
    private static class ExtFsInputStream extends FSInputStream {

      //The underlying data input stream that the
      // underlying filesystem will return.
      private FSDataInputStream underLyingStream;
      private byte[] oneBytebuff = new byte[1];
      private int nextLocation;
      private DistributedRaidFileSystem lfs;
      private Path path;
      private final DecodeInfo[] alternates;
      private final int buffersize;
      private final Configuration conf;
      private final int stripeLength;

      ExtFsInputStream(Configuration conf, DistributedRaidFileSystem lfs,
          DecodeInfo[] alternates, Path path, int stripeLength, int buffersize)
          throws IOException {
        this.underLyingStream = lfs.fs.open(path, buffersize);
        this.path = path;
        this.nextLocation = 0;
        this.alternates = alternates;
        this.buffersize = buffersize;
        this.conf = conf;
        this.lfs = lfs;
        this.stripeLength = stripeLength;
      }
      
      @Override
      public synchronized int available() throws IOException {
        int value = underLyingStream.available();
        nextLocation = 0;
        return value;
      }
      
      @Override
      public synchronized  void close() throws IOException {
        underLyingStream.close();
        super.close();
      }
      
      @Override
      public void mark(int readLimit) {
        underLyingStream.mark(readLimit);
        nextLocation = 0;
      }
      
      @Override
      public void reset() throws IOException {
        underLyingStream.reset();
        nextLocation = 0;
      }
      
      @Override
      public synchronized int read() throws IOException {
        long pos = underLyingStream.getPos();
        while (true) {
          try {
            int value = underLyingStream.read();
            nextLocation = 0;
            return value;
          } catch (BlockMissingException e) {
            setAlternateLocations(e, pos);
          } catch (ChecksumException e) {
            setAlternateLocations(e, pos);
          }
        }
      }
      
      @Override
      public synchronized int read(byte[] b) throws IOException {
        long pos = underLyingStream.getPos();
        while (true) {
          try{
            int value = underLyingStream.read(b);
            nextLocation = 0;
            return value;
          } catch (BlockMissingException e) {
            setAlternateLocations(e, pos);
          } catch (ChecksumException e) {
            setAlternateLocations(e, pos);
          }
        }
      }

      @Override
      public synchronized int read(byte[] b, int offset, int len) 
        throws IOException {
        long pos = underLyingStream.getPos();
        while (true) {
          try{
            int value = underLyingStream.read(b, offset, len);
            nextLocation = 0;
            return value;
          } catch (BlockMissingException e) {
            setAlternateLocations(e, pos);
          } catch (ChecksumException e) {
            setAlternateLocations(e, pos);
          }
        }
      }
      
      @Override
      public synchronized int read(long position, byte[] b, int offset, int len) 
        throws IOException {
        long pos = underLyingStream.getPos();
        while (true) {
          try {
            int value = underLyingStream.read(position, b, offset, len);
            nextLocation = 0;
            return value;
          } catch (BlockMissingException e) {
            setAlternateLocations(e, pos);
          } catch (ChecksumException e) {
            setAlternateLocations(e, pos);
          }
        }
      }
      
      @Override
      public synchronized long skip(long n) throws IOException {
        long value = underLyingStream.skip(n);
        nextLocation = 0;
        return value;
      }
      
      @Override
      public synchronized long getPos() throws IOException {
        long value = underLyingStream.getPos();
        nextLocation = 0;
        return value;
      }
      
      @Override
      public synchronized void seek(long pos) throws IOException {
        underLyingStream.seek(pos);
        nextLocation = 0;
      }

      @Override
      public boolean seekToNewSource(long targetPos) throws IOException {
        boolean value = underLyingStream.seekToNewSource(targetPos);
        nextLocation = 0;
        return value;
      }
      
      /**
       * position readable again.
       */
      @Override
      public void readFully(long pos, byte[] b, int offset, int length) 
        throws IOException {
        long post = underLyingStream.getPos();
        while (true) {
          try {
            underLyingStream.readFully(pos, b, offset, length);
            nextLocation = 0;
            return;
          } catch (BlockMissingException e) {
            setAlternateLocations(e, post);
          } catch (ChecksumException e) {
            setAlternateLocations(e, pos);
          }
        }
      }
      
      @Override
      public void readFully(long pos, byte[] b) throws IOException {
        long post = underLyingStream.getPos();
        while (true) {
          try {
            underLyingStream.readFully(pos, b);
            nextLocation = 0;
            return;
          } catch (BlockMissingException e) {
            setAlternateLocations(e, post);
          } catch (ChecksumException e) {
            setAlternateLocations(e, pos);
          }
        }
      }

      /**
       * Extract good file from RAID
       * @param curpos curexp the current exception
       * @param curpos the position of the current operation to be retried
       * @throws IOException if all alternate locations are exhausted
       */
      private void setAlternateLocations(IOException curexp, long curpos) 
        throws IOException {
        while (alternates != null && nextLocation < alternates.length) {
          try {
            int idx = nextLocation++;
            long corruptOffset = underLyingStream.getPos();

            // Make sure we use DFS and not DistributedRaidFileSystem for unRaid.
            Configuration clientConf = new Configuration(conf);
            Class<?> clazz = conf.getClass("fs.raid.underlyingfs.impl",
                                                DistributedFileSystem.class);
            clientConf.set("fs.hdfs.impl", clazz.getName());
            // Disable caching so that a previously cached RaidDfs is not used.
            clientConf.setBoolean("fs.hdfs.impl.disable.cache", true);
            Path npath = RaidNode.unRaid(clientConf, path,
                         alternates[idx].destPath,
                         alternates[idx].createDecoder(),
                         stripeLength, corruptOffset);
            if (npath == null)
              continue;

            FileSystem fs1 = getUnderlyingFileSystem(conf);
            fs1.initialize(npath.toUri(), conf);
            LOG.info("Opening alternate path " + npath + " at offset " + curpos);
            FSDataInputStream fd = fs1.open(npath, buffersize);
            fd.seek(curpos);
            underLyingStream.close();
            underLyingStream = fd;
            lfs.fs = fs1;
            path = npath;
            return;
          } catch (Exception e) {
            LOG.info("Error in using alternate path " + path + ". " + e +
                     " Ignoring...");
          }
        }
        throw curexp;
      }

      /**
       * The name of the file system that is immediately below the
       * DistributedRaidFileSystem. This is specified by the
       * configuration parameter called fs.raid.underlyingfs.impl.
       * If this parameter is not specified in the configuration, then
       * the default class DistributedFileSystem is returned.
       * @param conf the configuration object
       * @return the filesystem object immediately below DistributedRaidFileSystem
       * @throws IOException if all alternate locations are exhausted
       */
      private FileSystem getUnderlyingFileSystem(Configuration conf) {
        Class<?> clazz = conf.getClass("fs.raid.underlyingfs.impl", DistributedFileSystem.class);
        FileSystem fs = (FileSystem)ReflectionUtils.newInstance(clazz, conf);
        return fs;
      }
    }
  
    /**
     * constructor for ext input stream.
     * @param fs the underlying filesystem
     * @param p the path in the underlying file system
     * @param buffersize the size of IO
     * @throws IOException
     */
    public ExtFSDataInputStream(Configuration conf, DistributedRaidFileSystem lfs,
      DecodeInfo[] alternates, Path  p, int stripeLength, int buffersize) throws IOException {
        super(new ExtFsInputStream(conf, lfs, alternates, p, stripeLength, buffersize));
    }
  }
}
