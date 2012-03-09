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

package org.apache.hadoop.ipc;

import java.net.InetSocketAddress;

import junit.framework.TestCase;

import org.apache.avro.AvroRemoteException;
import org.apache.avro.util.Utf8;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.net.NetUtils;

/** Unit tests for AvroRpc. */
public class TestAvroRpc extends TestCase {
  private static final String ADDRESS = "0.0.0.0";

  public static final Log LOG =
    LogFactory.getLog(TestAvroRpc.class);
  
  private static Configuration conf = new Configuration();

  int datasize = 1024*100;
  int numThreads = 50;

  public TestAvroRpc(String name) { super(name); }
	
  public static class TestImpl implements AvroTestProtocol {

    public void ping() {}
    
    public String echo(String value) { return value; }

    public int add(int v1, int v2) { return v1 + v2; }

    public int error() throws Problem {
      throw new Problem();
    }
  }

  public void testCalls() throws Exception {
    Configuration conf = new Configuration();
    RPC.setProtocolEngine(conf, AvroTestProtocol.class, AvroRpcEngine.class);
    Server server = RPC.getServer(AvroTestProtocol.class,
                                  new TestImpl(), ADDRESS, 0, conf);
    AvroTestProtocol proxy = null;
    try {
      server.start();

      InetSocketAddress addr = NetUtils.getConnectAddress(server);
      proxy =
        (AvroTestProtocol)RPC.getProxy(AvroTestProtocol.class, 0, addr, conf);
      
      proxy.ping();

      String echo = proxy.echo("hello world");
      assertEquals("hello world", echo);

      int intResult = proxy.add(1, 2);
      assertEquals(3, intResult);

      boolean caught = false;
      try {
        proxy.error();
      } catch (AvroRemoteException e) {
        if(LOG.isDebugEnabled()) {
          LOG.debug("Caught " + e);
        }
        caught = true;
      }
      assertTrue(caught);

    } finally {
      server.stop();
    }
  }

  public void testAvroSpecificRpc() throws Exception {
    Configuration conf = new Configuration();
    RPC.setProtocolEngine(conf, AvroSpecificTestProtocol.class, 
        AvroSpecificRpcEngine.class);
    Server server = RPC.getServer(AvroSpecificTestProtocol.class,
                                  new AvroSpecificTestProtocolImpl(), 
                                  ADDRESS, 0, conf);
    AvroSpecificTestProtocol proxy = null;
    try {
      server.start();

      InetSocketAddress addr = NetUtils.getConnectAddress(server);
      proxy =
        (AvroSpecificTestProtocol)RPC.getProxy(AvroSpecificTestProtocol.class, 
            0, addr, conf);
      
      CharSequence echo = proxy.echo("hello world");
      assertEquals("hello world", echo.toString());

      int intResult = proxy.add(1, 2);
      assertEquals(3, intResult);

    } finally {
      server.stop();
    }
  }
  
  public static class AvroSpecificTestProtocolImpl implements 
      AvroSpecificTestProtocol {

    @Override
    public int add(int arg1, int arg2) throws AvroRemoteException {
      return arg1 + arg2;
    }

    @Override
    public CharSequence echo(CharSequence msg) throws AvroRemoteException {
      return msg;
    }
    
  }
  
}
