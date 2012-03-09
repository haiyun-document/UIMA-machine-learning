package org.apache.hadoop.examples;

import java.io.IOException;
import java.util.StringTokenizer;
import iitb.CRF.*;
import iitb.Model.*;
import iitb.Utils.*;


import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;


public class CRFmap {
	
  



  public static class TokenizerMapper 
       extends Mapper<Object, Text, IntWritable, DoubleWritable>{
	  
	  String inName;
	    String outDir;
	    String baseDir="";
	    int nlabels=7;

	    String delimit=" \t"; // used to define token boundaries
	    String tagDelimit="|"; // seperator between tokens and tag number
	    String impDelimit="\n\r\f,.:;?![]'"; // delimiters to be retained for tagging
	    String groupDelimit=null;

	    boolean confuseSet[]=null;
	    boolean validate = false; 
	    String mapTagString = null;
	    String smoothType = "";

	    String modelArgs = "";
	    String featureArgs = "";
	    String modelGraphType = "naive";

	    LabelMap labelMap;
	    Options options;

	    CRF crfModel;
	    FeatureGenImpl featureGen;
	    public FeatureGenerator featureGenerator() {return featureGen;};

	  
	  void  allocModel() throws Exception {
			// add any code related to dependency/consistency amongst paramter
			// values here..
			if (modelGraphType.equals("semi-markov")) {
			    if (options.getInt("debugLvl") > 1) {
				Util.printDbg("Creating semi-markov model");
			    }
			    NestedFeatureGenImpl nfgen = new NestedFeatureGenImpl(nlabels,options);
			    featureGen = nfgen;
			    crfModel = new NestedCRF(featureGen.numStates(),nfgen,options);
			} else 
			{
			    featureGen = new FeatureGenImpl(modelGraphType, nlabels);
			    crfModel=new CRF(featureGen.numStates(),featureGen,options);
		        }
		    }
	  
	  
	  
    public void map(Object key, Text value, Context context
                    ) throws IOException, InterruptedException {
     

		//filename
		System.out.println("the data is in"+value.toString());
		DataCruncher.createRaw(value.toString(),tagDelimit);
		TrainData trainData = DataCruncher.readTagged(nlabels,value.toString(),value.toString(),delimit,tagDelimit,impDelimit,labelMap);
		AlphaNumericPreprocessor.preprocess(trainData,nlabels);

		try {
			allocModel();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			
		}
		try {
			featureGen.train(trainData);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		double featureWts[] = crfModel.train(trainData);
		for (int ii = 0; ii<featureWts.length; ii++)
		{
        context.write(new IntWritable(ii), new DoubleWritable(ii));
      }
    }
  }
  
  public static class IntSumReducer 
       extends Reducer<IntWritable,DoubleWritable,Text,Text> {
    

    public void reduce(IntWritable key, Iterable<DoubleWritable> values, 
                       Context context
                       ) throws IOException, InterruptedException {
      
      context.write(new Text(key.toString()), new Text(values.toString()));

    }
  }

  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
  
    Job job = new Job(conf, "word count");
    job.setJarByClass(CRFmap.class);
    job.setMapperClass(TokenizerMapper.class);
    job.setCombinerClass(IntSumReducer.class);
    job.setReducerClass(IntSumReducer.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(Text.class);
    job.setInputFormatClass(TextInputFormat.class);
    job.setOutputFormatClass(TextOutputFormat.class);
    FileInputFormat.addInputPath(job, new Path(args[0]));
    FileOutputFormat.setOutputPath(job, new Path(args[1]));
    System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
  }

