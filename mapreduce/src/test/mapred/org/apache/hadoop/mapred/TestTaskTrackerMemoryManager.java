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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;
import java.util.regex.Matcher;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.MRConfig;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.MRJobConfig;
import org.apache.hadoop.mapreduce.server.jobtracker.JTConfig;
import org.apache.hadoop.mapreduce.server.tasktracker.TTConfig;
import org.apache.hadoop.mapreduce.util.LinuxResourceCalculatorPlugin;
import org.apache.hadoop.mapreduce.util.ProcfsBasedProcessTree;
import org.apache.hadoop.mapreduce.SleepJob;
import org.apache.hadoop.mapreduce.util.TestProcfsBasedProcessTree;
import org.apache.hadoop.util.StringUtils;
import org.apache.hadoop.util.ToolRunner;

import org.junit.After;
import org.junit.Ignore;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 * Test class to verify memory management of tasks.
 */
public class TestTaskTrackerMemoryManager {

  private static final Log LOG =
      LogFactory.getLog(TestTaskTrackerMemoryManager.class);
  private static String TEST_ROOT_DIR = new Path(System.getProperty(
		    "test.build.data", "/tmp")).toString().replace(' ', '+');

  private MiniMRCluster miniMRCluster;

  private String taskOverLimitPatternString =
      "TaskTree \\[pid=[0-9]*,tipID=.*\\] is running beyond.*memory-limits. "
          + "Current usage : [0-9]*bytes. Limit : %sbytes. Killing task.";

  private void startCluster(JobConf conf)
      throws Exception {
    conf.set(JTConfig.JT_IPC_HANDLER_COUNT, "1");
    conf.set(TTConfig.TT_MAP_SLOTS, "1");
    conf.set(TTConfig.TT_REDUCE_SLOTS, "1");
    conf.set(TTConfig.TT_SLEEP_TIME_BEFORE_SIG_KILL, "0");
    miniMRCluster = new MiniMRCluster(1, "file:///", 1, null, null, conf);
  }

  @After
  public void tearDown() {
    if (miniMRCluster != null) {
      miniMRCluster.shutdown();
    }
  }

  private int runSleepJob(JobConf conf) throws Exception {
    String[] args = { "-m", "3", "-r", "1", "-mt", "3000", "-rt", "1000" };
    return ToolRunner.run(conf, new SleepJob(), args);
  }

  private void runAndCheckSuccessfulJob(JobConf conf)
      throws IOException {
    Pattern taskOverLimitPattern =
        Pattern.compile(String.format(taskOverLimitPatternString, "[0-9]*"));
    Matcher mat = null;

    // Start the job.
    int ret;
    try {
      ret = runSleepJob(conf);
    } catch (Exception e) {
      ret = 1;
    }

    // Job has to succeed
    assertTrue(ret == 0);

    JobClient jClient = new JobClient(conf);
    JobStatus[] jStatus = jClient.getAllJobs();
    JobStatus js = jStatus[0]; // Our only job
    RunningJob rj = jClient.getJob(js.getJobID());

    // All events
    TaskCompletionEvent[] taskComplEvents = rj.getTaskCompletionEvents(0);

    for (TaskCompletionEvent tce : taskComplEvents) {
      String[] diagnostics =
          rj.getTaskDiagnostics(tce.getTaskAttemptId());

      if (diagnostics != null) {
        for (String str : diagnostics) {
          mat = taskOverLimitPattern.matcher(str);
          // The error pattern shouldn't be there in any TIP's diagnostics
          assertFalse(mat.find());
        }
      }
    }
  }

  private boolean isProcfsBasedTreeAvailable() {
    try {
      if (!ProcfsBasedProcessTree.isAvailable()) {
        LOG.info("Currently ProcessTree has only one implementation "
            + "ProcfsBasedProcessTree, which is not available on this "
            + "system. Not testing");
        return false;
      }
    } catch (Exception e) {
      LOG.info(StringUtils.stringifyException(e));
      return false;
    }
    return true;
  }

  /**
   * Test for verifying that nothing is killed when memory management is
   * disabled on the TT, even when the tasks run over their limits.
   * 
   * @throws Exception
   */
  @Test
  public void testTTLimitsDisabled()
      throws Exception {
    // Run the test only if memory management is enabled
    if (!isProcfsBasedTreeAvailable()) {
      return;
    }

    // Task-memory management disabled by default.
    startCluster(new JobConf());
    long PER_TASK_LIMIT = 1L; // Doesn't matter how low.
    JobConf conf = miniMRCluster.createJobConf();
    conf.setMemoryForMapTask(PER_TASK_LIMIT);
    conf.setMemoryForReduceTask(PER_TASK_LIMIT);
    runAndCheckSuccessfulJob(conf);
  }

  /**
   * Test for verifying that tasks within limits, with the cumulative usage also
   * under TT's limits succeed.
   * 
   * @throws Exception
   */
  @Test
  public void testTasksWithinLimits()
      throws Exception {
    // Run the test only if memory management is enabled
    if (!isProcfsBasedTreeAvailable()) {
      return;
    }

    // Large so that sleepjob goes through and fits total TT usage
    long PER_TASK_LIMIT = 2 * 1024L;

    // Start cluster with proper configuration.
    JobConf fConf = new JobConf();
    fConf.setLong(MRConfig.MAPMEMORY_MB, 2 * 1024L);
    fConf.setLong(MRConfig.REDUCEMEMORY_MB, 2 * 1024L);
    // Reserve only 1 mb of the memory on TaskTrackers
    fConf.setLong(TTConfig.TT_RESERVED_PHYSCIALMEMORY_MB, 1L);
    startCluster(new JobConf());

    JobConf conf = new JobConf(miniMRCluster.createJobConf());
    conf.setMemoryForMapTask(PER_TASK_LIMIT);
    conf.setMemoryForReduceTask(PER_TASK_LIMIT);
    // Set task physical memory limits
    conf.setLong(MRJobConfig.MAP_MEMORY_PHYSICAL_MB, PER_TASK_LIMIT);
    conf.setLong(MRJobConfig.REDUCE_MEMORY_PHYSICAL_MB, PER_TASK_LIMIT);
    runAndCheckSuccessfulJob(conf);
  }

  /**
   * Test for verifying that tasks that go beyond limits get killed.
   * 
   * @throws Exception
   */
  @Ignore("Intermittent, unexpected task success causes test to fail.")
  @Test
  public void testTasksBeyondLimits()
      throws Exception {

    // Run the test only if memory management is enabled
    if (!isProcfsBasedTreeAvailable()) {
      return;
    }

    // Start cluster with proper configuration.
    JobConf fConf = new JobConf();
    // very small value, so that no task escapes to successful completion.
    fConf.setInt(TTConfig.TT_MEMORY_MANAGER_MONITORING_INTERVAL, 100);
    fConf.setLong(MRConfig.MAPMEMORY_MB, 2 * 1024);
    fConf.setLong(MRConfig.REDUCEMEMORY_MB, 2 * 1024);
    startCluster(fConf);
    runJobExceedingMemoryLimit(false);
  }
  
  /**
   * Test for verifying that tasks that go beyond physical limits get killed.
   * 
   * @throws Exception
   */
  @Ignore("Intermittent, unexpected task success causes test to fail.")
  @Test
  public void testTasksBeyondPhysicalLimits()
      throws Exception {

    // Run the test only if memory management is enabled
    if (!isProcfsBasedTreeAvailable()) {
      return;
    }

    // Start cluster with proper configuration.
    JobConf fConf = new JobConf();
    // very small value, so that no task escapes to successful completion.
    fConf.setInt(TTConfig.TT_MEMORY_MANAGER_MONITORING_INTERVAL, 100);
    // Reserve only 1 mb of the memory on TaskTrackers
    fConf.setLong(TTConfig.TT_RESERVED_PHYSCIALMEMORY_MB, 1L);
    startCluster(fConf);
    runJobExceedingMemoryLimit(true);
  }
  
  /**
   * Runs tests with tasks beyond limit and using old configuration values for
   * the TaskTracker.
   * 
   * @throws Exception
   */
  @Ignore("Intermittent, unexpected task success causes test to fail.")
  @Test
  public void testTaskMemoryMonitoringWithDeprecatedConfiguration () 
    throws Exception {
    
    // Run the test only if memory management is enabled
    if (!isProcfsBasedTreeAvailable()) {
      return;
    }
    // Start cluster with proper configuration.
    JobConf fConf = new JobConf();
    // very small value, so that no task escapes to successful completion.
    fConf.setInt(TTConfig.TT_MEMORY_MANAGER_MONITORING_INTERVAL, 100);
    //set old values, max vm property per task and upper limit on the tasks
    //vm
    //setting the default maximum vmem property to 2 GB
    fConf.setLong(JobConf.MAPRED_TASK_DEFAULT_MAXVMEM_PROPERTY,
        (2L * 1024L * 1024L * 1024L));
    fConf.setLong(JobConf.UPPER_LIMIT_ON_TASK_VMEM_PROPERTY, 
        (3L * 1024L * 1024L * 1024L));
    startCluster(fConf);
    runJobExceedingMemoryLimit(false);
  }

  /**
   * Runs a job which should fail the when run by the memory monitor.
   * 
   * @param doPhysicalMemory If it is true, use physical memory limit.
   *                         Otherwise use virtual memory limit.
   * @throws IOException
   */
  private void runJobExceedingMemoryLimit(boolean doPhysicalMemory)
    throws IOException {
    long PER_TASK_LIMIT = 1L; // Low enough to kill off sleepJob tasks.

    Pattern taskOverLimitPattern =
        Pattern.compile(String.format(taskOverLimitPatternString, String
            .valueOf(PER_TASK_LIMIT*1024*1024L)));
    Matcher mat = null;

    // Set up job.
    JobConf conf = new JobConf(miniMRCluster.createJobConf());
    if (doPhysicalMemory) {
      conf.setLong(MRJobConfig.MAP_MEMORY_PHYSICAL_MB, PER_TASK_LIMIT);
      conf.setLong(MRJobConfig.REDUCE_MEMORY_PHYSICAL_MB, PER_TASK_LIMIT);
    } else {
      conf.setMemoryForMapTask(PER_TASK_LIMIT);
      conf.setMemoryForReduceTask(PER_TASK_LIMIT);
    }
    conf.setMaxMapAttempts(1);
    conf.setMaxReduceAttempts(1);

    // Start the job.
    int ret = 0;
    try {
      ret = runSleepJob(conf);
    } catch (Exception e) {
      ret = 1;
    }

    // Job has to fail
    assertTrue(ret != 0);

    JobClient jClient = new JobClient(conf);
    JobStatus[] jStatus = jClient.getAllJobs();
    JobStatus js = jStatus[0]; // Our only job
    RunningJob rj = jClient.getJob(js.getJobID());

    // All events
    TaskCompletionEvent[] taskComplEvents = rj.getTaskCompletionEvents(0);

    for (TaskCompletionEvent tce : taskComplEvents) {
      // Every task HAS to fail
      assertTrue("Failure expected, task: " + tce.getTaskStatus(),
          tce.getTaskStatus() == TaskCompletionEvent.Status.TIPFAILED ||
          tce.getTaskStatus() == TaskCompletionEvent.Status.FAILED);

      String[] diagnostics =
          rj.getTaskDiagnostics(tce.getTaskAttemptId());

      // Every task HAS to spit out the out-of-memory errors
      assertNotNull(diagnostics);

      for (String str : diagnostics) {
        mat = taskOverLimitPattern.matcher(str);
        // Every task HAS to spit out the out-of-memory errors in the same
        // format. And these are the only diagnostic messages.
        assertTrue(mat.find());
      }
    }
  }

  /**
   * Test for verifying that tasks causing cumulative usage to go beyond TT's
   * limit get killed even though they all are under individual limits. Memory
   * management for tasks with disabled task-limits also traverses the same
   * code-path, so we don't need a separate testTaskLimitsDisabled.
   * 
   * @throws Exception
   */
  @Test
  public void testTasksCumulativelyExceedingTTLimits()
      throws Exception {

    // Run the test only if memory management is enabled
    if (!isProcfsBasedTreeAvailable()) {
      return;
    }

    // Large enough for SleepJob Tasks.
    long PER_TASK_LIMIT = 100 * 1024L;

    // Start cluster with proper configuration.
    JobConf fConf = new JobConf();
    fConf.setLong(MRConfig.MAPMEMORY_MB,
        1L);
    fConf.setLong(
        MRConfig.REDUCEMEMORY_MB, 1L);

    // Because of the above, the total tt limit is 2mb
    long TASK_TRACKER_LIMIT = 2 * 1024 * 1024L;

    // very small value, so that no task escapes to successful completion.
    fConf.setInt(TTConfig.TT_MEMORY_MANAGER_MONITORING_INTERVAL, 100);

    startCluster(fConf);

    Pattern taskOverLimitPattern =
      Pattern.compile(String.format(taskOverLimitPatternString, String
          .valueOf(PER_TASK_LIMIT)));

    Pattern trackerOverLimitPattern =
      Pattern
          .compile("Killing one of the least progress tasks - .*, as "
              + "the cumulative memory usage of all the tasks on the TaskTracker"
              + " exceeds virtual memory limit " + TASK_TRACKER_LIMIT + ".");
    Matcher mat = null;

    // Set up job.
    JobConf conf = new JobConf(miniMRCluster.createJobConf());
    conf.setMemoryForMapTask(PER_TASK_LIMIT);
    conf.setMemoryForReduceTask(PER_TASK_LIMIT);

    JobClient jClient = new JobClient(conf);
    SleepJob sleepJob = new SleepJob();
    sleepJob.setConf(conf);
    // Start the job
    Job job = sleepJob.createJob(1, 1, 5000, 1, 1000, 1);
    job.submit();
    boolean TTOverFlowMsgPresent = false;
    while (true) {
      List<TaskReport> allTaskReports = new ArrayList<TaskReport>();
      allTaskReports.addAll(Arrays.asList(jClient
          .getSetupTaskReports(JobID.downgrade(job.getJobID()))));
      allTaskReports.addAll(Arrays.asList(jClient
          .getMapTaskReports(JobID.downgrade(job.getJobID()))));
      for (TaskReport tr : allTaskReports) {
        String[] diag = tr.getDiagnostics();
        for (String str : diag) {
          mat = taskOverLimitPattern.matcher(str);
          assertFalse(mat.find());
          mat = trackerOverLimitPattern.matcher(str);
          if (mat.find()) {
            TTOverFlowMsgPresent = true;
          }
        }
      }
      if (TTOverFlowMsgPresent) {
        break;
      }
      try {
        Thread.sleep(1000);
      } catch (InterruptedException e) {
        // nothing
      }
    }
    // If it comes here without a test-timeout, it means there was a task that
    // was killed because of crossing cumulative TT limit.

    // Test succeeded, kill the job.
    job.killJob();
  }
  
  /**
   * Test to verify the check for whether a process tree is over limit or not.
   * @throws IOException if there was a problem setting up the
   *                      fake procfs directories or files.
   */
  @Test
  public void testProcessTreeLimits() throws IOException {
    
    // set up a dummy proc file system
    File procfsRootDir = new File(TEST_ROOT_DIR, "proc");
    String[] pids = { "100", "200", "300", "400", "500", "600", "700" };
    try {
      TestProcfsBasedProcessTree.setupProcfsRootDir(procfsRootDir);
      
      // create pid dirs.
      TestProcfsBasedProcessTree.setupPidDirs(procfsRootDir, pids);
      
      // create process infos.
      TestProcfsBasedProcessTree.ProcessStatInfo[] procs =
          new TestProcfsBasedProcessTree.ProcessStatInfo[7];

      // assume pids 100, 500 are in 1 tree 
      // 200,300,400 are in another
      // 600,700 are in a third
      procs[0] = new TestProcfsBasedProcessTree.ProcessStatInfo(
          new String[] {"100", "proc1", "1", "100", "100", "100000"});
      procs[1] = new TestProcfsBasedProcessTree.ProcessStatInfo(
          new String[] {"200", "proc2", "1", "200", "200", "200000"});
      procs[2] = new TestProcfsBasedProcessTree.ProcessStatInfo(
          new String[] {"300", "proc3", "200", "200", "200", "300000"});
      procs[3] = new TestProcfsBasedProcessTree.ProcessStatInfo(
          new String[] {"400", "proc4", "200", "200", "200", "400000"});
      procs[4] = new TestProcfsBasedProcessTree.ProcessStatInfo(
          new String[] {"500", "proc5", "100", "100", "100", "1500000"});
      procs[5] = new TestProcfsBasedProcessTree.ProcessStatInfo(
          new String[] {"600", "proc6", "1", "600", "600", "100000"});
      procs[6] = new TestProcfsBasedProcessTree.ProcessStatInfo(
          new String[] {"700", "proc7", "600", "600", "600", "100000"});
      // write stat files.
      TestProcfsBasedProcessTree.writeStatFiles(procfsRootDir, pids, procs);

      // vmem limit
      long limit = 700000;
      
      // Create TaskMemoryMonitorThread
      TaskMemoryManagerThread test = new TaskMemoryManagerThread(1000000L,
                                                                5000L);
      // create process trees
      // tree rooted at 100 is over limit immediately, as it is
      // twice over the mem limit.
      ProcfsBasedProcessTree pTree = new ProcfsBasedProcessTree(
                                          "100", true, 100L, 
                                          procfsRootDir.getAbsolutePath());
      pTree.getProcessTree();
      assertTrue("tree rooted at 100 should be over limit " +
                    "after first iteration.",
                  test.isProcessTreeOverLimit(pTree, "dummyId", limit));
      
      // the tree rooted at 200 is initially below limit.
      pTree = new ProcfsBasedProcessTree("200", true, 100L,
                                          procfsRootDir.getAbsolutePath());
      pTree.getProcessTree();
      assertFalse("tree rooted at 200 shouldn't be over limit " +
                    "after one iteration.",
                  test.isProcessTreeOverLimit(pTree, "dummyId", limit));
      // second iteration - now the tree has been over limit twice,
      // hence it should be declared over limit.
      pTree.getProcessTree();
      assertTrue("tree rooted at 200 should be over limit after 2 iterations",
                  test.isProcessTreeOverLimit(pTree, "dummyId", limit));
      
      // the tree rooted at 600 is never over limit.
      pTree = new ProcfsBasedProcessTree("600", true, 100L,
                                            procfsRootDir.getAbsolutePath());
      pTree.getProcessTree();
      assertFalse("tree rooted at 600 should never be over limit.",
                    test.isProcessTreeOverLimit(pTree, "dummyId", limit));
      
      // another iteration does not make any difference.
      pTree.getProcessTree();
      assertFalse("tree rooted at 600 should never be over limit.",
                    test.isProcessTreeOverLimit(pTree, "dummyId", limit));
    } finally {
      FileUtil.fullyDelete(procfsRootDir);
    }
  }

  /**
   * Test for verifying that tasks causing cumulative usage of physical memory
   * to go beyond TT's limit get killed.
   *
   * @throws Exception
   */
  @Test
  public void testTasksCumulativelyExceedingTTPhysicalLimits()
      throws Exception {

    // Run the test only if memory management is enabled
    if (!isProcfsBasedTreeAvailable()) {
      return;
    }

    // Start cluster with proper configuration.
    JobConf fConf = new JobConf();

    // very small value, so that no task escapes to successful completion.
    fConf.setInt(TTConfig.TT_MEMORY_MANAGER_MONITORING_INTERVAL, 100);
    
    // reserve all memory on TT so that the job will exceed memory limits
    LinuxResourceCalculatorPlugin memoryCalculatorPlugin =
            new LinuxResourceCalculatorPlugin();
    long totalPhysicalMemory = memoryCalculatorPlugin.getPhysicalMemorySize();
    long reservedPhysicalMemory = totalPhysicalMemory / (1024 * 1024) + 1;
    fConf.setLong(TTConfig.TT_RESERVED_PHYSCIALMEMORY_MB,
                  reservedPhysicalMemory);
    long maxRssMemoryAllowedForAllTasks = totalPhysicalMemory -
                                          reservedPhysicalMemory * 1024 * 1024L;
    Pattern physicalMemoryOverLimitPattern = Pattern.compile(
        "Killing one of the memory-consuming tasks - .*"
          + ", as the cumulative RSS memory usage of all the tasks on "
          + "the TaskTracker exceeds physical memory limit "
          + maxRssMemoryAllowedForAllTasks + ".");

    startCluster(fConf);
    Matcher mat = null;

    // Set up job.
    JobConf conf = new JobConf(miniMRCluster.createJobConf());
    // Set per task physical memory limits to be a higher value
    conf.setLong(MRJobConfig.MAP_MEMORY_PHYSICAL_MB, 2 * 1024L);
    conf.setLong(MRJobConfig.REDUCE_MEMORY_PHYSICAL_MB, 2 * 1024L);
    JobClient jClient = new JobClient(conf);
    SleepJob sleepJob = new SleepJob();
    sleepJob.setConf(conf);
    // Start the job
    Job job = sleepJob.createJob(1, 1, 100000, 1, 100000, 1);
    job.submit();
    boolean TTOverFlowMsgPresent = false;
    while (true) {
      List<TaskReport> allTaskReports = new ArrayList<TaskReport>();
      allTaskReports.addAll(Arrays.asList(jClient
          .getSetupTaskReports(JobID.downgrade(job.getJobID()))));
      allTaskReports.addAll(Arrays.asList(jClient
          .getMapTaskReports(JobID.downgrade(job.getJobID()))));
      for (TaskReport tr : allTaskReports) {
        String[] diag = tr.getDiagnostics();
        for (String str : diag) {
          mat = physicalMemoryOverLimitPattern.matcher(str);
          if (mat.find()) {
            TTOverFlowMsgPresent = true;
          }
        }
      }
      if (TTOverFlowMsgPresent) {
        break;
      }
      assertFalse("Job should not finish successfully", job.isSuccessful());
      try {
        Thread.sleep(1000);
      } catch (InterruptedException e) {
        // nothing
      }
    }
    // If it comes here without a test-timeout, it means there was a task that
    // was killed because of crossing cumulative TT limit.

    // Test succeeded, kill the job.
    job.killJob();
  }
}
