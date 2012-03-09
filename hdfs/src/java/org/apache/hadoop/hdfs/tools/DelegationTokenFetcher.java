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
package org.apache.hadoop.hdfs.tools;

import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLConnection;
import java.security.PrivilegedExceptionAction;
import java.util.Collection;
import java.util.Date;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.classification.InterfaceAudience;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hdfs.DistributedFileSystem;
import org.apache.hadoop.hdfs.HdfsConfiguration;
import org.apache.hadoop.hdfs.security.token.delegation.DelegationTokenIdentifier;
import org.apache.hadoop.hdfs.security.token.delegation.DelegationTokenSecretManager;
import org.apache.hadoop.hdfs.server.namenode.CancelDelegationTokenServlet;
import org.apache.hadoop.hdfs.server.namenode.GetDelegationTokenServlet;
import org.apache.hadoop.hdfs.server.namenode.RenewDelegationTokenServlet;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.security.Credentials;
import org.apache.hadoop.security.SecurityUtil;
import org.apache.hadoop.security.UserGroupInformation;
import org.apache.hadoop.security.token.Token;
import org.apache.hadoop.util.GenericOptionsParser;

/**
 * Fetch a DelegationToken from the current Namenode and store it in the
 * specified file.
 */
@InterfaceAudience.Private
public class DelegationTokenFetcher {
  private static final Log LOG = 
    LogFactory.getLog(DelegationTokenFetcher.class);
  private static final String WEBSERVICE = "webservice";
  private static final String RENEWER = "renewer";
  private static final String CANCEL = "cancel";
  private static final String RENEW = "renew";
  private static final String PRINT = "print";

  static {
    // Enable Kerberos sockets
    System.setProperty("https.cipherSuites", "TLS_KRB5_WITH_3DES_EDE_CBC_SHA");
  }

  private static void printUsage(PrintStream err) throws IOException {
    err.println("fetchdt retrieves delegation tokens from the NameNode");
    err.println();
    err.println("fetchdt <opts> <token file>");
    err.println("Options:");
    err.println("  --webservice <url>  Url to contact NN on");
    err.println("  --renewer <name>    Name of the delegation token renewer");
    err.println("  --cancel            Cancel the delegation token");
    err.println("  --renew             Renew the delegation token");
    err.println("  --print             Print the delegation token");
    err.println();
    GenericOptionsParser.printGenericCommandUsage(err);
    System.exit(1);
  }

  private static Collection<Token<?>> readTokens(Path file, Configuration conf)
      throws IOException {
    Credentials creds = Credentials.readTokenStorageFile(file, conf);
    return creds.getAllTokens();
  }
    
  /**
   * Command-line interface
   */
  public static void main(final String[] args) throws Exception {
    final Configuration conf = new HdfsConfiguration();
    Options fetcherOptions = new Options();
    fetcherOptions.addOption(WEBSERVICE, true,
        "HTTPS url to reach the NameNode at");
    fetcherOptions.addOption(RENEWER, true,
        "Name of the delegation token renewer");
    fetcherOptions.addOption(CANCEL, false, "cancel the token");
    fetcherOptions.addOption(RENEW, false, "renew the token");
    fetcherOptions.addOption(PRINT, false, "print the token");
    GenericOptionsParser parser = new GenericOptionsParser(conf,
        fetcherOptions, args);
    CommandLine cmd = parser.getCommandLine();
    
    // get options
    final String webUrl = cmd.hasOption(WEBSERVICE) ? cmd
        .getOptionValue(WEBSERVICE) : null;
    final String renewer = cmd.hasOption(RENEWER) ? 
        cmd.getOptionValue(RENEWER) : null;
    final boolean cancel = cmd.hasOption(CANCEL);
    final boolean renew = cmd.hasOption(RENEW);
    final boolean print = cmd.hasOption(PRINT);
    String[] remaining = parser.getRemainingArgs();

    // check option validity
    if (cancel && renew || cancel && print || renew && print || cancel && renew
        && print) {
      System.err.println("ERROR: Only specify cancel, renew or print.");
      printUsage(System.err);
    }
    if (remaining.length != 1 || remaining[0].charAt(0) == '-') {
      System.err.println("ERROR: Must specify exacltly one token file");
      printUsage(System.err);
    }
    // default to using the local file system
    FileSystem local = FileSystem.getLocal(conf);
    final Path tokenFile = new Path(local.getWorkingDirectory(), remaining[0]);

    // Login the current user
    UserGroupInformation.getCurrentUser().doAs(
        new PrivilegedExceptionAction<Object>() {
          @SuppressWarnings("unchecked")
          @Override
          public Object run() throws Exception {

            if (print) {
              DelegationTokenIdentifier id = new DelegationTokenSecretManager(
                  0, 0, 0, 0, null).createIdentifier();
              for (Token<?> token : readTokens(tokenFile, conf)) {
                DataInputStream in = new DataInputStream(
                    new ByteArrayInputStream(token.getIdentifier()));
                id.readFields(in);
                System.out.println("Token (" + id + ") for " + token.getService());
              }
              return null;
            }
            
            if (webUrl != null) {
              if (renew) {
                long result;
                for (Token<?> token : readTokens(tokenFile, conf)) {
                  result = renewDelegationToken(webUrl,
                      (Token<DelegationTokenIdentifier>) token);
                  System.out.println("Renewed token via " + webUrl + " for "
                      + token.getService() + " until: " + new Date(result));
                }
              } else if (cancel) {
                for (Token<?> token : readTokens(tokenFile, conf)) {
                  cancelDelegationToken(webUrl,
                      (Token<DelegationTokenIdentifier>) token);
                  System.out.println("Cancelled token via " + webUrl + " for "
                      + token.getService());
                }
              } else {
                Credentials creds = getDTfromRemote(webUrl, renewer);
                creds.writeTokenStorageFile(tokenFile, conf);
                for (Token<?> token : creds.getAllTokens()) {
                  System.out.println("Fetched token via " + webUrl + " for "
                      + token.getService() + " into " + tokenFile);
                }
              }
            } else {
              FileSystem fs = FileSystem.get(conf);
              if (cancel) {
                for (Token<?> token : readTokens(tokenFile, conf)) {
                  ((DistributedFileSystem) fs)
                      .cancelDelegationToken((Token<DelegationTokenIdentifier>) token);
                  System.out.println("Cancelled token for "
                      + token.getService());
                }
              } else if (renew) {
                long result;
                for (Token<?> token : readTokens(tokenFile, conf)) {
                  result = ((DistributedFileSystem) fs)
                      .renewDelegationToken((Token<DelegationTokenIdentifier>) token);
                  System.out.println("Renewed token for " + token.getService()
                      + " until: " + new Date(result));
                }
              } else {
                Token<?> token = fs.getDelegationToken(renewer);
                Credentials cred = new Credentials();
                cred.addToken(token.getService(), token);
                cred.writeTokenStorageFile(tokenFile, conf);
                System.out.println("Fetched token for " + token.getService()
                    + " into " + tokenFile);
              }
            }
            return null;
          }
        });
  }
  
  static public Credentials getDTfromRemote(String nnAddr, 
      String renewer) throws IOException {
    DataInputStream dis = null;

    try {
      StringBuffer url = new StringBuffer();
      if (renewer != null) {
        url.append(nnAddr).append(GetDelegationTokenServlet.PATH_SPEC).append("?").
        append(GetDelegationTokenServlet.RENEWER).append("=").append(renewer);
      } else {
        url.append(nnAddr).append(GetDelegationTokenServlet.PATH_SPEC);
      }
      URL remoteURL = new URL(url.toString());
      SecurityUtil.fetchServiceTicket(remoteURL);
      URLConnection connection = remoteURL.openConnection();

      InputStream in = connection.getInputStream();
      Credentials ts = new Credentials();
      dis = new DataInputStream(in);
      ts.readFields(dis);
      return ts;
    } catch (Exception e) {
      throw new IOException("Unable to obtain remote token", e);
    } finally {
      if(dis != null) dis.close();
    }
  }

  /**
   * Renew a Delegation Token.
   * @param nnAddr the NameNode's address
   * @param tok the token to renew
   * @return the Date that the token will expire next.
   * @throws IOException
   */
  static public long renewDelegationToken(String nnAddr,
      Token<DelegationTokenIdentifier> tok
  ) throws IOException {
    StringBuilder buf = new StringBuilder();
    buf.append(nnAddr);
    buf.append(RenewDelegationTokenServlet.PATH_SPEC);
    buf.append("?");
    buf.append(RenewDelegationTokenServlet.TOKEN);
    buf.append("=");
    buf.append(tok.encodeToUrlString());
    BufferedReader in = null;
    try {
      URL url = new URL(buf.toString());
      SecurityUtil.fetchServiceTicket(url);
      HttpURLConnection connection = (HttpURLConnection) url.openConnection();
      if (connection.getResponseCode() != HttpURLConnection.HTTP_OK) {
        throw new IOException("Error renewing token: " + 
            connection.getResponseMessage());
      }
      in = new BufferedReader(new InputStreamReader
          (connection.getInputStream()));
      long result = Long.parseLong(in.readLine());
      in.close();
      return result;
    } catch (IOException ie) {
      IOUtils.cleanup(LOG, in);
      throw ie;
    }
  }

  /**
   * Cancel a Delegation Token.
   * @param nnAddr the NameNode's address
   * @param tok the token to cancel
   * @throws IOException
   */
  static public void cancelDelegationToken(String nnAddr,
      Token<DelegationTokenIdentifier> tok
  ) throws IOException {
    StringBuilder buf = new StringBuilder();
    buf.append(nnAddr);
    buf.append(CancelDelegationTokenServlet.PATH_SPEC);
    buf.append("?");
    buf.append(CancelDelegationTokenServlet.TOKEN);
    buf.append("=");
    buf.append(tok.encodeToUrlString());
    BufferedReader in = null;
    try {
      URL url = new URL(buf.toString());
      SecurityUtil.fetchServiceTicket(url);
      HttpURLConnection connection = (HttpURLConnection) url.openConnection();
      if (connection.getResponseCode() != HttpURLConnection.HTTP_OK) {
        throw new IOException("Error cancelling token: " + 
            connection.getResponseMessage());
      }
    } catch (IOException ie) {
      IOUtils.cleanup(LOG, in);
      throw ie;
    }
  }
}
