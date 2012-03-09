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

import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;

import org.apache.hadoop.fs.FSInputStream;
import org.apache.hadoop.hdfs.server.namenode.StreamFile;


/**
 * To support HTTP byte streams, a new connection to an HTTP server needs to be
 * created each time. This class hides the complexity of those multiple 
 * connections from the client. Whenever seek() is called, a new connection
 * is made on the successive read(). The normal input stream functions are 
 * connected to the currently active input stream. 
 */
class ByteRangeInputStream extends FSInputStream {
  
  /**
   * This class wraps a URL to allow easy mocking when testing. The URL class
   * cannot be easily mocked because it is public.
   */
  static class URLOpener {
    protected URL url;
  
    public URLOpener(URL u) {
      url = u;
    }
  
    public void setURL(URL u) {
      url = u;
    }
  
    public URL getURL() {
      return url;
    }
  
    public HttpURLConnection openConnection() throws IOException {
      return (HttpURLConnection)url.openConnection();
    }  
  }
  
  
  protected InputStream in;
  protected URLOpener originalURL;
  protected URLOpener resolvedURL;
  protected long startPos = 0;
  protected long currentPos = 0;
  protected long filelength;

  protected int status = STATUS_SEEK;
  protected static final int STATUS_NORMAL = 0;
  protected static final int STATUS_SEEK = 1;

  ByteRangeInputStream(final URL url) {
    this(new URLOpener(url), new URLOpener(null));
  }
  
  ByteRangeInputStream(URLOpener o, URLOpener r) {
    this.originalURL = o;
    this.resolvedURL = r;
  }
  
  private InputStream getInputStream() throws IOException {
    if (status != STATUS_NORMAL) {
      
      if (in != null) {
        in.close();
        in = null;
      }
      
      // use the original url  if no resolved url exists (e.g., if it's 
      // the first time a request is made)
      final URLOpener o = resolvedURL.getURL() == null? originalURL: resolvedURL;

      final HttpURLConnection connection = o.openConnection();
      try {
        connection.setRequestMethod("GET");
        if (startPos != 0) {
          connection.setRequestProperty("Range", "bytes="+startPos+"-");
        }
        connection.connect();
        final String cl = connection.getHeaderField(StreamFile.CONTENT_LENGTH);
        filelength = cl == null? -1: Long.parseLong(cl);
        if (HftpFileSystem.LOG.isDebugEnabled()) {
          HftpFileSystem.LOG.debug("filelength = " + filelength);
        }
        in = connection.getInputStream();
      } catch(IOException ioe) {
        HftpFileSystem.throwIOExceptionFromConnection(connection, ioe);
      }
      
      if (startPos != 0 && connection.getResponseCode() != 206) {
        // we asked for a byte range but did not receive a partial content
        // response...
        throw new IOException("206 expected, but received "
                              + connection.getResponseCode());
      } else if(startPos == 0 && connection.getResponseCode() != 200) {
        // we asked for all bytes from the beginning but didn't receive a 200
        // response (none of the other 2xx codes are valid here)
        throw new IOException("200 expected, but received "
                              + connection.getResponseCode());
      }
      
      resolvedURL.setURL(connection.getURL());
      status = STATUS_NORMAL;
    }
    
    return in;
  }
  
  private void update(final boolean isEOF, final int n
      ) throws IOException {
    if (!isEOF) {
      currentPos += n;
    } else if (currentPos < filelength) {
      throw new IOException("Got EOF but currentPos = " + currentPos
          + " < filelength = " + filelength);
    }
  }

  public int read() throws IOException {
    final int b = getInputStream().read();
    update(b == -1, 1);
    return b;
  }
  
  /**
   * Seek to the given offset from the start of the file.
   * The next read() will be from that location.  Can't
   * seek past the end of the file.
   */
  public void seek(long pos) throws IOException {
    if (pos != currentPos) {
      startPos = pos;
      currentPos = pos;
      status = STATUS_SEEK;
    }
  }

  /**
   * Return the current offset from the start of the file
   */
  public long getPos() throws IOException {
    return currentPos; // keep total count?
  }

  /**
   * Seeks a different copy of the data.  Returns true if
   * found a new source, false otherwise.
   */
  public boolean seekToNewSource(long targetPos) throws IOException {
    return false;
  }

}


