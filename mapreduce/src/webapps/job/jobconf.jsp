<%
/*
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
%>
<%@ page
  contentType="text/html; charset=UTF-8"
  import="javax.servlet.*"
  import="javax.servlet.http.*"
  import="java.io.*"
  import="java.net.URL"
  import="org.apache.hadoop.mapred.*"
  import="org.apache.hadoop.mapred.JSPUtil.JobWithViewAccessCheck"
  import="org.apache.hadoop.util.*"
%>
<%!	private static final long serialVersionUID = 1L;
%>


<%
  final JobTracker tracker = (JobTracker) application.getAttribute("job.tracker");
  final String jobId = request.getParameter("jobid");
  if (jobId == null) {
    out.println("<h2>Missing 'jobid' for fetching job configuration!</h2>");
    return;
  }
%>
  
<html>

<title>Job Configuration: JobId - <%= jobId %></title>

<body>
<h2>Job Configuration: JobId - <%= jobId %></h2><br>

<%

  final JobID jobIdObj = JobID.forName(jobId);
  JobWithViewAccessCheck myJob = JSPUtil.checkAccessAndGetJob(tracker,
      jobIdObj, request, response);
  if (!myJob.isViewJobAllowed()) {
    return; // user is not authorized to view this job
  }

  JobInProgress job = myJob.getJob();
  // redirect to history page if it cannot be found in memory
  if (job == null) {
    String historyFile = tracker.getJobHistory().getHistoryFilePath(jobIdObj);
    if (historyFile == null) {
      out.println("<h2>Job " + jobId + " not known!</h2>");
      return;
    }
    String historyUrl = "/jobconf_history.jsp?logFile=" + historyFile;
    response.sendRedirect(response.encodeRedirectURL(historyUrl));
    return;
  }

  String jobFilePath = tracker.getLocalJobFilePath(JobID.forName(jobId));
  FileInputStream jobFile = null;
  try {
    jobFile = new FileInputStream(jobFilePath);
    JobConf jobConf = new JobConf(jobFilePath);
    XMLUtils.transform(
        jobConf.getConfResourceAsInputStream("webapps/static/jobconf.xsl"),
        jobFile, out);
  } catch (Exception e) {
    out.println("Failed to retreive job configuration for job '" + jobId + "!");
    out.println(e);
  } finally {
    if (jobFile != null) {
      try { 
        jobFile.close(); 
      } catch (IOException e) {}
    }
  }
%>

<br>
<%
out.println(ServletUtil.htmlFooter());
%>
