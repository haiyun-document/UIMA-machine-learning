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

import java.io.File;
import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.mapred.UtilsForTests.FakeClock;
import org.apache.hadoop.mapreduce.MRConfig;
import org.apache.hadoop.mapreduce.MRJobConfig;
import org.apache.hadoop.mapreduce.server.tasktracker.Localizer;
import org.apache.hadoop.mapreduce.util.MRAsyncDiskService;

import static org.junit.Assert.*;

import org.junit.After;
import org.junit.Test;

public class TestUserLogCleanup {
  private static String jtid = "test";
  private static long ONE_HOUR = 1000 * 60 * 60;
  private Localizer localizer;
  private UserLogCleaner taskLogCleanupThread;
  private TaskTracker tt;
  private FakeClock myClock = new FakeClock();
  private JobID jobid1 = new JobID(jtid, 1);
  private JobID jobid2 = new JobID(jtid, 2);
  private JobID jobid3 = new JobID(jtid, 3);
  private JobID jobid4 = new JobID(jtid, 4);
  private File foo = new File(TaskLog.getUserLogDir(), "foo");
  private File bar = new File(TaskLog.getUserLogDir(), "bar");

  public TestUserLogCleanup() throws IOException {
    Configuration conf = new Configuration();
    localizer = new Localizer(FileSystem.get(conf), conf
        .getStrings(MRConfig.LOCAL_DIR), new DefaultTaskController());
    taskLogCleanupThread = new UserLogCleaner(conf);
    taskLogCleanupThread.setClock(myClock);
    tt = new TaskTracker();
    tt.setConf(new JobConf(conf));
    tt.setLocalizer(localizer);
    tt.setTaskLogCleanupThread(taskLogCleanupThread);
  }

  @After
  public void tearDown() throws IOException {
    FileUtil.fullyDelete(TaskLog.getUserLogDir());
  }

  private File localizeJob(JobID jobid) throws IOException {
    File jobUserlog = TaskLog.getJobDir(jobid);

    JobConf conf = new JobConf();
    // localize job log directory
    tt.initializeJobLogDir(jobid, conf);
    assertTrue(jobUserlog + " directory is not created.", jobUserlog.exists());
    return jobUserlog;
  }

  private void jobFinished(JobID jobid, int logRetainHours) {
    Configuration jobconf = new Configuration();
    jobconf.setInt(MRJobConfig.USER_LOG_RETAIN_HOURS, logRetainHours);
    taskLogCleanupThread.markJobLogsForDeletion(myClock.getTime(), jobconf,
        jobid);
  }

  /**
   * Tests job user-log directory deletion.
   * 
   * Adds two jobs for log deletion. One with one hour retain hours, other with
   * two retain hours. After an hour,
   * TaskLogCleanupThread.processCompletedJobs() call,
   * makes sure job with 1hr retain hours is removed and other is retained.
   * After one more hour, job with 2hr retain hours is also removed.
   * 
   * @throws IOException
   */
  @Test
  public void testJobLogCleanup() throws IOException {
    File jobUserlog1 = localizeJob(jobid1);
    File jobUserlog2 = localizeJob(jobid2);

    // add job user log directory for deletion, with 2 hours for deletion
    jobFinished(jobid1, 2);

    // add the job for deletion with one hour as retain hours
    jobFinished(jobid2, 1);

    // remove old logs and see jobid1 is not removed and jobid2 is removed
    myClock.advance(ONE_HOUR);
    taskLogCleanupThread.processCompletedJobs();
    assertTrue(jobUserlog1 + " got deleted", jobUserlog1.exists());
    assertFalse(jobUserlog2 + " still exists.", jobUserlog2.exists());

    myClock.advance(ONE_HOUR);
    // remove old logs and see jobid1 is removed now
    taskLogCleanupThread.processCompletedJobs();
    assertFalse(jobUserlog1 + " still exists.", jobUserlog1.exists());
  }

  /**
   * Tests user-log directory cleanup on a TT re-init with 3 hours as log
   * retain hours for tracker. 
   * 
   * Adds job1 deletion before the re-init with 2 hour retain hours. 
   * Adds job2 for which there are no tasks/killJobAction after the re-init.
   * Adds job3 for which there is localizeJob followed by killJobAction 
   * with 3 hours as retain hours.
   * Adds job4 for which there are some tasks after the re-init.
   * 
   * @throws IOException
   */
  @Test
  public void testUserLogCleanup() throws IOException {
    File jobUserlog1 = localizeJob(jobid1);
    File jobUserlog2 = localizeJob(jobid2);
    File jobUserlog3 = localizeJob(jobid3);
    File jobUserlog4 = localizeJob(jobid4);
    // create a some files/dirs in userlog
    foo.mkdirs();
    bar.createNewFile();

    // add the jobid1 for deletion with retainhours = 2
    jobFinished(jobid1, 2);

    // time is now 1.
    myClock.advance(ONE_HOUR);
    
    // mimic TaskTracker reinit
    // clear userlog directory
    // job directories will be added with 3 hours as retain hours. They will be
    // deleted at time 4.
    Configuration conf = new Configuration();
    conf.setInt(MRJobConfig.USER_LOG_RETAIN_HOURS, 3);
    taskLogCleanupThread.clearOldUserLogs(conf);
    assertFalse(foo.exists());
    assertFalse(bar.exists());
    assertTrue(jobUserlog1.exists());
    assertTrue(jobUserlog2.exists());
    assertTrue(jobUserlog3.exists());
    assertTrue(jobUserlog4.exists());
    assertTrue(new File(TaskLog.getUserLogDir(), MRAsyncDiskService.TOBEDELETED)
        .exists());

    myClock.advance(ONE_HOUR);
    // time is now 2.
    taskLogCleanupThread.processCompletedJobs();
    assertFalse(jobUserlog1.exists());
    assertTrue(jobUserlog2.exists());
    assertTrue(jobUserlog3.exists());
    assertTrue(jobUserlog4.exists());
    
    // mimic localizeJob followed KillJobAction for jobid3
    // add the job for deletion with retainhours = 3. 
    // jobid3 should be deleted at time 5.
    jobUserlog3 = localizeJob(jobid3);
    jobFinished(jobid3, 3);

    // mimic localizeJob for jobid4
    jobUserlog4 = localizeJob(jobid4);

    // do cleanup
    myClock.advance(2 * ONE_HOUR);
    // time is now 4.
    taskLogCleanupThread.processCompletedJobs();

    // jobid2 will be deleted
    assertFalse(jobUserlog1.exists());
    assertFalse(jobUserlog2.exists());
    assertTrue(jobUserlog3.exists());
    assertTrue(jobUserlog4.exists());

    myClock.advance(ONE_HOUR);
    // time is now 5.
    // do cleanup again
    taskLogCleanupThread.processCompletedJobs();

    // jobid3 will be deleted
    assertFalse(jobUserlog1.exists());
    assertFalse(jobUserlog2.exists());
    assertFalse(jobUserlog3.exists());
    assertTrue(jobUserlog4.exists());
  }

  /**
   * Tests user-log directory cleanup on a TT restart.
   * 
   * Adds job1 deletion before the restart with 2 hour retain hours.
   * Adds job2 for which there are no tasks/killJobAction after the restart.
   * Adds job3 for which there is localizeJob followed by killJobAction after
   * the restart with 3 hours retain hours.
   * Adds job4 for which there are some tasks after the restart.
   * 
   * @throws IOException
   */
  @Test
  public void testUserLogCleanupAfterRestart() throws IOException {
    File jobUserlog1 = localizeJob(jobid1);
    File jobUserlog2 = localizeJob(jobid2);
    File jobUserlog3 = localizeJob(jobid3);
    File jobUserlog4 = localizeJob(jobid4);
    // create a some files/dirs in userlog
    foo.mkdirs();
    bar.createNewFile();

    // add the jobid1 for deletion with retainhours = 2
    jobFinished(jobid1, 2);

    // time is now 1.
    myClock.advance(ONE_HOUR);
    
    // mimic TaskTracker restart
    // clear userlog directory
    // job directories will be added with 3 hours as retain hours. 
    Configuration conf = new Configuration();
    conf.setInt(MRJobConfig.USER_LOG_RETAIN_HOURS, 3);
    taskLogCleanupThread = new UserLogCleaner(conf);
    myClock = new FakeClock(); // clock is reset.
    taskLogCleanupThread.setClock(myClock);
    taskLogCleanupThread.clearOldUserLogs(conf);
    tt.setTaskLogCleanupThread(taskLogCleanupThread);
    assertFalse(foo.exists());
    assertFalse(bar.exists());
    assertTrue(jobUserlog1.exists());
    assertTrue(jobUserlog2.exists());
    assertTrue(jobUserlog3.exists());
    assertTrue(jobUserlog4.exists());
    assertTrue(new File(TaskLog.getUserLogDir(), MRAsyncDiskService.TOBEDELETED)
        .exists());

    myClock.advance(ONE_HOUR);
    // time is now 1.
    taskLogCleanupThread.processCompletedJobs();
    assertTrue(jobUserlog1.exists());
    assertTrue(jobUserlog2.exists());
    assertTrue(jobUserlog3.exists());
    assertTrue(jobUserlog4.exists());
    
    // mimic localizeJob followed KillJobAction for jobid3
    // add the job for deletion with retainhours = 3. 
    // jobid3 should be deleted at time 4.
    jobUserlog3 = localizeJob(jobid3);
    jobFinished(jobid3, 3);

    // mimic localizeJob for jobid4
    jobUserlog4 = localizeJob(jobid4);

    // do cleanup
    myClock.advance(2 * ONE_HOUR);
    // time is now 3.
    taskLogCleanupThread.processCompletedJobs();

    // jobid1 and jobid2 will be deleted
    assertFalse(jobUserlog1.exists());
    assertFalse(jobUserlog2.exists());
    assertTrue(jobUserlog3.exists());
    assertTrue(jobUserlog4.exists());

    myClock.advance(ONE_HOUR);
    // time is now 4.
    // do cleanup again
    taskLogCleanupThread.processCompletedJobs();

    // jobid3 will be deleted
    assertFalse(jobUserlog1.exists());
    assertFalse(jobUserlog2.exists());
    assertFalse(jobUserlog3.exists());
    assertTrue(jobUserlog4.exists());
  }
}
