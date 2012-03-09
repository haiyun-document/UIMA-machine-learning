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
package org.apache.hadoop.mapred;

import org.apache.hadoop.mapreduce.server.jobtracker.JTConfig;
import org.apache.hadoop.security.UserGroupInformation;
import org.apache.hadoop.mapreduce.SleepJob;
import org.apache.hadoop.util.ToolRunner;
import org.apache.hadoop.conf.Configuration;
import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.net.HttpURLConnection;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.TimeUnit;
import java.util.ArrayList;
import java.util.List;

import org.junit.Test;
import org.junit.BeforeClass;
import org.junit.AfterClass;
import static org.junit.Assert.*;

/**
 * System tests for the fair scheduler. These run slower than the
 * mock-based tests in TestFairScheduler but have a better chance
 * of catching synchronization bugs with the real JT.
 *
 * This test suite will often be run inside JCarder in order to catch
 * deadlock bugs which have plagued the scheduler in the past - hence
 * it is a bit of a "grab-bag" of system tests, since it's important
 * that they all run as part of the same JVM instantiation.
 */
public class TestFairSchedulerSystem {
  static final int NUM_THREADS=2;

  static MiniMRCluster mr;
  static JobConf conf;

  @BeforeClass
  public static void setUp() throws Exception {
    conf = new JobConf();
    final int taskTrackers = 1;

    // Bump up the frequency of preemption updates to test against
    // deadlocks, etc.
    conf.set("mapred.jobtracker.taskScheduler", FairScheduler.class.getCanonicalName());
    conf.set("mapred.fairscheduler.update.interval", "1");
    conf.set("mapred.fairscheduler.preemption.interval", "1");
    conf.set("mapred.fairscheduler.preemption", "true");
    conf.set("mapred.fairscheduler.eventlog.enabled", "true");
    conf.set("mapred.fairscheduler.poolnameproperty", "group.name");
    conf.set(JTConfig.JT_PERSIST_JOBSTATUS, "false");
    mr = new MiniMRCluster(taskTrackers, "file:///", 1, null, null, conf);
  }

  @AfterClass
  public static void tearDown() throws Exception {
    if (mr != null) {
      mr.shutdown();
    }
  }

  private void runSleepJob(JobConf conf) throws Exception {
    String[] args = { "-m", "1", "-r", "1", "-mt", "1", "-rt", "1" };
    ToolRunner.run(conf, new SleepJob(), args);
  }

  /**
   * Submit some concurrent sleep jobs, and visit the scheduler servlet
   * while they're running.
   */
  @Test
  public void testFairSchedulerSystem() throws Exception {
    ExecutorService exec = Executors.newFixedThreadPool(NUM_THREADS);
    List<Future<Void>> futures = new ArrayList<Future<Void>>(NUM_THREADS);
    for (int i = 0; i < NUM_THREADS; i++) {
      futures.add(exec.submit(new Callable<Void>() {
            public Void call() throws Exception {
              JobConf jobConf = mr.createJobConf();
              runSleepJob(jobConf);
              return null;
            }
          }));
    }

    // Wait for the tasks to finish, and visit the scheduler servlet
    // every few seconds while waiting.
    for (Future<Void> future : futures) {
      while (true) {
        try {
          future.get(3, TimeUnit.SECONDS);
          break;
        } catch (TimeoutException te) {
          // It's OK
        }
        checkServlet(true);
        checkServlet(false);
      }
    }
  }

  /**
   * Check the fair scheduler servlet for good status code and smoke test
   * for contents.
   */
  private void checkServlet(boolean advanced) throws Exception {
    String jtURL = "http://localhost:" +
      mr.getJobTrackerRunner().getJobTrackerInfoPort();
    URL url = new URL(jtURL + "/scheduler" +
                      (advanced ? "?advanced" : ""));
    HttpURLConnection connection = (HttpURLConnection)url.openConnection();
    connection.setRequestMethod("GET");
    connection.connect();
    assertEquals(200, connection.getResponseCode());

    // Just to be sure, slurp the content and make sure it looks like the scheduler
    BufferedReader reader = new BufferedReader(
      new InputStreamReader(connection.getInputStream()));
    StringBuilder sb = new StringBuilder();

    String line = null;
    while ((line = reader.readLine()) != null) {
      sb.append(line).append('\n');
    }

    String contents = sb.toString();
    assertTrue(contents.contains("Fair Scheduler Administration"));


    String userGroups[] = UserGroupInformation.getCurrentUser().getGroupNames();
    String primaryGroup = ">" + userGroups[0] + "<";
    assertTrue(contents.contains(primaryGroup));
 }
}
