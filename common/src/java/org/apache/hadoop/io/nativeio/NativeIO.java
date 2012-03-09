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
package org.apache.hadoop.io.nativeio;

import java.io.FileDescriptor;
import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.util.NativeCodeLoader;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
/**
 * JNI wrappers for various native IO-related calls not available in Java.
 * These functions should generally be used alongside a fallback to another
 * more portable mechanism.
 */
public class NativeIO {
  // Flags for open() call from bits/fcntl.h
  public static final int O_RDONLY   =    00;
  public static final int O_WRONLY   =    01;
  public static final int O_RDWR     =    02;
  public static final int O_CREAT    =  0100;
  public static final int O_EXCL     =  0200;
  public static final int O_NOCTTY   =  0400;
  public static final int O_TRUNC    = 01000;
  public static final int O_APPEND   = 02000;
  public static final int O_NONBLOCK = 04000;
  public static final int O_SYNC   =  010000;
  public static final int O_ASYNC  =  020000;
  public static final int O_FSYNC = O_SYNC;
  public static final int O_NDELAY = O_NONBLOCK;

  private static final Log LOG = LogFactory.getLog(NativeIO.class);

  private static boolean nativeLoaded = false;
  private static boolean workaroundNonThreadSafePasswdCalls = false;

  static final String WORKAROUND_NON_THREADSAFE_CALLS_KEY =
    "hadoop.workaround.non.threadsafe.getpwuid";
  static final boolean WORKAROUND_NON_THREADSAFE_CALLS_DEFAULT = false;

  static {
    if (NativeCodeLoader.isNativeCodeLoaded()) {
      try {
        Configuration conf = new Configuration();
        workaroundNonThreadSafePasswdCalls = conf.getBoolean(
          WORKAROUND_NON_THREADSAFE_CALLS_KEY,
          WORKAROUND_NON_THREADSAFE_CALLS_DEFAULT);

        initNative();
        nativeLoaded = true;
      } catch (Throwable t) {
        // This can happen if the user has an older version of libhadoop.so
        // installed - in this case we can continue without native IO
        // after warning
        LOG.error("Unable to initialize NativeIO libraries", t);
      }
    }
  }

  /**
   * Return true if the JNI-based native IO extensions are available.
   */
  public static boolean isAvailable() {
    return NativeCodeLoader.isNativeCodeLoaded() && nativeLoaded;
  }

  /** Wrapper around open(2) */
  public static native FileDescriptor open(String path, int flags, int mode) throws IOException;
  /** Wrapper around fstat(2) */
  public static native Stat fstat(FileDescriptor fd) throws IOException;
  /** Wrapper around chmod(2) */
  public static native void chmod(String path, int mode) throws IOException;

  /** Initialize the JNI method ID and class ID cache */
  private static native void initNative();


  /**
   * Result type of the fstat call
   */
  public static class Stat {
    private String owner, group;
    private int mode;

    // Mode constants
    public static final int S_IFMT = 0170000;      /* type of file */
    public static final int   S_IFIFO  = 0010000;  /* named pipe (fifo) */
    public static final int   S_IFCHR  = 0020000;  /* character special */
    public static final int   S_IFDIR  = 0040000;  /* directory */
    public static final int   S_IFBLK  = 0060000;  /* block special */
    public static final int   S_IFREG  = 0100000;  /* regular */
    public static final int   S_IFLNK  = 0120000;  /* symbolic link */
    public static final int   S_IFSOCK = 0140000;  /* socket */
    public static final int   S_IFWHT  = 0160000;  /* whiteout */
    public static final int S_ISUID = 0004000;  /* set user id on execution */
    public static final int S_ISGID = 0002000;  /* set group id on execution */
    public static final int S_ISVTX = 0001000;  /* save swapped text even after use */
    public static final int S_IRUSR = 0000400;  /* read permission, owner */
    public static final int S_IWUSR = 0000200;  /* write permission, owner */
    public static final int S_IXUSR = 0000100;  /* execute/search permission, owner */

    Stat(String owner, String group, int mode) {
      this.owner = owner;
      this.group = group;
      this.mode = mode;
    }

    public String toString() {
      return "Stat(owner='" + owner + "', group='" + group + "'" +
        ", mode=" + mode + ")";
    }

    public String getOwner() {
      return owner;
    }
    public String getGroup() {
      return group;
    }
    public int getMode() {
      return mode;
    }
  }
}
