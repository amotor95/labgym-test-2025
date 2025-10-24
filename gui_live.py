'''
Copyright (C)
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see https://tldrlegal.com/license/gnu-general-public-license-v3-(gpl-3)#fulltext. 

For license issues, please contact:

Dr. Bing Ye
Life Sciences Institute
University of Michigan
210 Washtenaw Avenue, Room 5403
Ann Arbor, MI 48109-2216
USA

Email: bingye@umich.edu
'''



import multiprocessing.process
import matplotlib.pyplot as plt
import wx
import os
import platform
import shutil
import matplotlib as mpl
from pathlib import Path
import pandas as pd
import torch
import json
import pathlib
from .analyzebehavior import AnalyzeAnimal
from .analyzebehavior_dt import AnalyzeAnimalDetector
from .tools import plot_events,parse_all_events_file,calculate_distances
from .minedata import data_mining

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math
import numpy as np

import sys
sys.path.append('/opt/anaconda3/envs/labgym/lib/python3.10/site-packages/LabGym')

# Have to navigate to labgym folder and run __main__ with python
from LabGym.labgym_transformer import save_model, load_model, training, predict, validate, BehaviorTransformer

from LabGym.labgym_encoder import LabGym_Encoder

import cv2, time

import threading

from ordered_dict import ThreadSafeOrderedDict

import queue

import multiprocessing

# For choose categorizer function
the_absolute_current_path=str(Path(__file__).resolve().parent)

class ColorPicker(wx.Dialog):

	'''
	A window for select a color for each behavior
	'''

	def __init__(self,parent,title,name_and_color):

		super(ColorPicker,self).__init__(parent=None,title=title,size=(200,200))

		self.name_and_color=name_and_color
		name=self.name_and_color[0]
		hex_color=self.name_and_color[1][1].lstrip('#')
		color=tuple(int(hex_color[i:i+2],16) for i in (0,2,4))

		boxsizer=wx.BoxSizer(wx.VERTICAL)

		self.color_picker=wx.ColourPickerCtrl(self,colour=color)

		button=wx.Button(self,wx.ID_OK,label='Apply')

		boxsizer.Add(0,10,0)
		boxsizer.Add(self.color_picker,0,wx.ALL|wx.CENTER,10)
		boxsizer.Add(button,0,wx.ALL|wx.CENTER,10)
		boxsizer.Add(0,10,0)

		self.SetSizer(boxsizer)

class Train_Worker:
	def __init__(self, status, config, output_dir, num_processed_clips, worker_id):
		# Dynamically assign all config keys as attributes
		for key, value in config.items():
			setattr(self, key, value)
		device = "cuda" if torch.cuda.is_available() else "cpu"
		# print(f"Using {device} device")
		self.status = status
		# The most recently processed clip across all workers
		self.num_processed_clips = num_processed_clips
		self.worker_id = worker_id
		self.output_dir = output_dir
		print("Worker " + str(worker_id) + " initialized!")

	def copy_to_output_folder(self, clip_results_folder, clip_filename):
		"""Appends new behavior analysis results to a master Excel file, adjusting times to be continuous."""
		new_data_file = os.path.join(clip_results_folder, "all_events.xlsx")

		output_copy_path = os.path.join(self.output_dir, clip_filename+"_events.xlsx")

		os.makedirs(self.output_dir, exist_ok=True)

		shutil.copy(new_data_file, output_copy_path)
	
	def __call__(self):
			while not self.status["analysis_queue"].empty():
				print("!!!!Queue Size: ", self.status["analysis_queue"].qsize())
				task = self.status["analysis_queue"].get()
				clip_filename = task["clip_filename"]
				clip_folder = task["clip_folder"]
				print("Processing clip #", clip_filename)
				# Make clip path
				clip_path = pathlib.Path(clip_folder, clip_filename)
				# Make folder to hold clip folders
				analyzed_clips_folder = pathlib.Path(self.output_dir, "analyzed_clips")
				analyzed_clips_folder.mkdir(parents=True, exist_ok=True)
				# Make clip folder
				current_clip_folder = pathlib.Path(analyzed_clips_folder, clip_filename)
				current_clip_folder.mkdir(parents=True, exist_ok=True)
				# Set clip path to clip and result path to clip folder (DON'T COPY CLIP)
				self.path_to_videos = [clip_path]
				self.result_path = str(current_clip_folder)

				self.analyze_behaviors(event=None)

				# Append results to a master spreadsheet
				self.copy_to_output_folder(current_clip_folder, clip_filename)

				self.num_processed_clips.value += 1

				print("Finished processing clip #", clip_filename)
	
	def analyze_behaviors(self, event):
		# Replaced all instances of self.path_to_videos and self.result_path with the path_to_videos and result_path arguments
		# Was messing up with threading because they were trying to overwrite shared instance paths

		if self.path_to_videos is None or self.result_path is None:

			wx.MessageBox('No input video(s) / result folder.','Error',wx.OK|wx.ICON_ERROR)

		else:

			if self.behavior_mode==3:

				if self.path_to_categorizer is None or self.path_to_detector is None:
					wx.MessageBox('You need to select a Categorizer / Detector.','Error',wx.OK|wx.ICON_ERROR)
				else:
					if len(self.animal_to_include)==0:
						self.animal_to_include=self.animal_kinds
					if self.detector_batch<=0:
						self.detector_batch=1
					if self.behavior_to_include[0]=='all':
						self.behavior_to_include=list(self.behaviornames_and_colors.keys())
					AAD=AnalyzeAnimalDetector()
					AAD.analyze_images_individuals(self.path_to_detector,self.path_to_videos,self.result_path,self.animal_kinds,path_to_categorizer=self.path_to_categorizer,
						generate=False,animal_to_include=self.animal_to_include,behavior_to_include=self.behavior_to_include,names_and_colors=self.behaviornames_and_colors,
						imagewidth=self.framewidth,dim_conv=self.dim_conv,channel=self.channel,detection_threshold=self.detection_threshold,uncertain=self.uncertain,
						background_free=self.background_free,black_background=self.black_background,social_distance=0)

			else:

				all_events={}
				event_data={}
				all_time=[]

				if self.use_detector:
					for animal_name in self.animal_kinds:
						all_events[animal_name]={}
					if len(self.animal_to_include)==0:
						self.animal_to_include=self.animal_kinds
					if self.detector_batch<=0:
						self.detector_batch=1

				if self.path_to_categorizer is None:
					self.behavior_to_include=[]
				else:
					if self.behavior_to_include[0]=='all':
						self.behavior_to_include=list(self.behaviornames_and_colors.keys())

				for i in self.path_to_videos:

					filename=os.path.splitext(os.path.basename(i))[0].split('_')
					if self.decode_animalnumber:
						if self.use_detector:
							self.animal_number={}
							number=[x[1:] for x in filename if len(x)>1 and x[0]=='n']
							for a,animal_name in enumerate(self.animal_kinds):
								self.animal_number[animal_name]=int(number[a])
						else:
							for x in filename:
								if len(x)>1:
									if x[0]=='n':
										self.animal_number=int(x[1:])
					if self.decode_t:
						for x in filename:
							if len(x)>1:
								if x[0]=='b':
									self.t=float(x[1:])
					if self.decode_extraction:
						for x in filename:
							if len(x)>2:
								if x[:2]=='xs':
									self.ex_start=int(x[2:])
								if x[:2]=='xe':
									self.ex_end=int(x[2:])

					if self.animal_number is None:
						if self.use_detector:
							self.animal_number={}
							for animal_name in self.animal_kinds:
								self.animal_number[animal_name]=1
						else:
							self.animal_number=1
				
					if self.path_to_categorizer is None:
						self.behavior_mode=0
						categorize_behavior=False
					else:
						categorize_behavior=True

					if self.use_detector is False:

						AA=AnalyzeAnimal()
						AA.prepare_analysis(i,self.result_path,self.animal_number,delta=self.delta,names_and_colors=self.behaviornames_and_colors,
							framewidth=self.framewidth,stable_illumination=self.stable_illumination,dim_tconv=self.dim_tconv,dim_conv=self.dim_conv,channel=self.channel,
							include_bodyparts=self.include_bodyparts,std=self.std,categorize_behavior=categorize_behavior,animation_analyzer=self.animation_analyzer,
							path_background=self.background_path,autofind_t=self.autofind_t,t=self.t,duration=self.duration,ex_start=self.ex_start,ex_end=self.ex_end,
							length=self.length,animal_vs_bg=self.animal_vs_bg)
						if self.behavior_mode==0:
							AA.acquire_information(background_free=self.background_free,black_background=self.black_background)
							AA.craft_data()
							interact_all=False
						else:
							AA.acquire_information_interact_basic(background_free=self.background_free,black_background=self.black_background)
							interact_all=True
						if self.path_to_categorizer is not None:
							AA.categorize_behaviors(self.path_to_categorizer,uncertain=self.uncertain,min_length=self.min_length)
						AA.annotate_video(self.behavior_to_include,show_legend=self.show_legend,interact_all=interact_all)
						AA.export_results(normalize_distance=self.normalize_distance,parameter_to_analyze=self.parameter_to_analyze)

						if self.path_to_categorizer is not None:
							for n in AA.event_probability:
								all_events[len(all_events)]=AA.event_probability[n]
							if len(all_time)<len(AA.all_time):
								all_time=AA.all_time

					else:

						AAD=AnalyzeAnimalDetector()
						AAD.prepare_analysis(self.path_to_detector,i,self.result_path,self.animal_number,self.animal_kinds,self.behavior_mode,
							names_and_colors=self.behaviornames_and_colors,framewidth=self.framewidth,dim_tconv=self.dim_tconv,dim_conv=self.dim_conv,channel=self.channel,
							include_bodyparts=self.include_bodyparts,std=self.std,categorize_behavior=categorize_behavior,animation_analyzer=self.animation_analyzer,
							t=self.t,duration=self.duration,length=self.length,social_distance=self.social_distance)
						if self.behavior_mode==1:
							AAD.acquire_information_interact_basic(batch_size=self.detector_batch,background_free=self.background_free,black_background=self.black_background)
						else:
							AAD.acquire_information(batch_size=self.detector_batch,background_free=self.background_free,black_background=self.black_background)
						if self.behavior_mode!=1:
							AAD.craft_data()
						if self.path_to_categorizer is not None:
							AAD.categorize_behaviors(self.path_to_categorizer,uncertain=self.uncertain,min_length=self.min_length)
						if self.correct_ID:
							AAD.correct_identity(self.specific_behaviors)
						AAD.annotate_video(self.animal_to_include,self.behavior_to_include,show_legend=self.show_legend)
						AAD.export_results(normalize_distance=self.normalize_distance,parameter_to_analyze=self.parameter_to_analyze)

						if self.path_to_categorizer is not None:
							for animal_name in self.animal_kinds:
								for n in AAD.event_probability[animal_name]:
									all_events[animal_name][len(all_events[animal_name])]=AAD.event_probability[animal_name][n]
							if len(all_time)<len(AAD.all_time):
								all_time=AAD.all_time
					
				if self.path_to_categorizer is not None:

					max_length=len(all_time)

					if self.use_detector is False:

						for n in all_events:
							event_data[len(event_data)]=all_events[n]+[['NA',-1]]*(max_length-len(all_events[n]))
						all_events_df=pd.DataFrame(event_data,index=all_time)
						all_events_df.to_excel(os.path.join(self.result_path,'all_events.xlsx'),float_format='%.2f',index_label='time/ID')
						plot_events(self.result_path,event_data,all_time,self.behaviornames_and_colors,self.behavior_to_include,width=0,height=0)
						folders=[i for i in os.listdir(self.result_path) if os.path.isdir(os.path.join(self.result_path,i))]
						folders.sort()
						for behavior_name in self.behaviornames_and_colors:
							all_summary=[]
							for folder in folders:
								individual_summary=os.path.join(self.result_path,folder,behavior_name,'all_summary.xlsx')
								if os.path.exists(individual_summary):
									all_summary.append(pd.read_excel(individual_summary))
							if len(all_summary)>=1:
								all_summary=pd.concat(all_summary,ignore_index=True)
								all_summary.to_excel(os.path.join(self.result_path,behavior_name+'_summary.xlsx'),float_format='%.2f',index_label='ID/parameter')

					else:

						for animal_name in self.animal_to_include:
							for n in all_events[animal_name]:
								event_data[len(event_data)]=all_events[animal_name][n]+[['NA',-1]]*(max_length-len(all_events[animal_name][n]))
							event_data[len(event_data)]=[['NA',-1]]*max_length
						del event_data[len(event_data)-1]
						all_events_df=pd.DataFrame(event_data,index=all_time)
						all_events_df.to_excel(os.path.join(self.result_path,'all_events.xlsx'),float_format='%.2f',index_label='time/ID')
						plot_events(self.result_path,event_data,all_time,self.behaviornames_and_colors,self.behavior_to_include,width=0,height=0)
						folders=[i for i in os.listdir(self.result_path) if os.path.isdir(os.path.join(self.result_path,i))]
						folders.sort()
						for animal_name in self.animal_kinds:
							for behavior_name in self.behaviornames_and_colors:
								all_summary=[]
								for folder in folders:
									individual_summary=os.path.join(self.result_path,folder,behavior_name,animal_name+'_all_summary.xlsx')
									if os.path.exists(individual_summary):
										all_summary.append(pd.read_excel(individual_summary))
								if len(all_summary)>=1:
									all_summary=pd.concat(all_summary,ignore_index=True)
									all_summary.to_excel(os.path.join(self.result_path,animal_name+'_'+behavior_name+'_summary.xlsx'),float_format='%.2f',index_label='ID/parameter')

			print('Analysis completed!')

class WindowLv2_TrainTransformer(wx.Frame):

	'''
	The 'Train Transformer' window
	'''

	def __init__(self,title):

		super(WindowLv2_TrainTransformer,self).__init__(parent=None,title=title,size=(1000,760))
		self.path_to_analysis_results=None # the file that stores LabGym analysis results
		self.out_path=None # the folder to store the transformer model
		self.embedding_dimensions = None # num of embedding dimensions for the transformer (how many features)
		self.num_heads = None # number of transformer attention heads
		self.src_len = None # transformer input length
		self.tgt_len = None # transformer output length
		self.data_input_mode = 'video' # data mode
		self.data_input_path = None # folder to training data
		self.num_workers = 1 # number of workers if analyzing videos for training data

		#COPY
		self.behavior_mode=0 # 0--non-interactive, 1--interactive basic, 2--interactive advanced, 3--static images
		self.use_detector=False # whether the Detector is used
		self.detector_path=None # the 'LabGym/detectors' folder, which stores all the trained Detectors
		self.path_to_detector=None # path to the Detector
		self.detector_batch=1 # for batch processing use if GPU is available
		self.detection_threshold=0 # only for 'static images' behavior mode
		self.animal_kinds=[] # the total categories of animals / objects in a Detector
		self.background_path=None # if not None, load background images from path in 'background subtraction' detection method
		self.model_path=None # the 'LabGym/models' folder, which stores all the trained Categorizers
		self.path_to_categorizer=None # path to the Categorizer
		self.path_to_videos=None # path to a batch of videos for analysis
		self.result_path=None # the folder for storing analysis outputs
		self.framewidth=None # if not None, will resize the video frame keeping the original w:h ratio
		self.delta=10000 # the fold changes in illumination that determines the optogenetic stimulation onset
		self.decode_animalnumber=False # whether to decode animal numbers from '_nn_' in video file names
		self.animal_number=1 # the number of animals / objects in a video
		self.autofind_t=False # whether to find stimulation onset automatically (only for optogenetics)
		self.decode_t=False # whether to decode start_t from '_bt_' in video file names
		self.t=0 # the start_t for analysis
		self.duration=0 # the duration of the analysis
		self.decode_extraction=False # whether to decode time windows for background extraction from '_xst_' and '_xet_' in video file names
		self.ex_start=0 # start time for background extraction
		self.ex_end=None # end time for background extraction
		self.behaviornames_and_colors={} # behavior names in the Categorizer and their representative colors for annotation
		self.dim_tconv=8 # input dimension for Animation Analyzer in Categorizer
		self.dim_conv=8 # input dimension for Pattern Recognizer in Categorizer
		self.channel=1 # input channel for Animation Analyzer, 1--gray scale, 3--RGB scale
		self.length=15 # input time step for Animation Analyzer, also the duration / length for a behavior example
		self.animal_vs_bg=0 # 0: animals birghter than the background; 1: animals darker than the background; 2: hard to tell
		self.stable_illumination=True # whether the illumination in videos is stable
		self.animation_analyzer=True # whether to include Animation Analyzer in the Categorizers
		self.animal_to_include=[] # the animals / obejcts that will be annotated in the annotated videos / behavior plots
		self.behavior_to_include=['all'] # behaviors that will be annotated in the annotated videos / behavior plots
		self.parameter_to_analyze=[] # quantitative measures that will be included in the quantification
		self.include_bodyparts=False # whether to include body parts in the pattern images
		self.std=0 # a value between 0 and 255, higher value, less body parts will be included in the pattern images
		self.uncertain=0 # a threshold between the highest the 2nd highest probablity of behaviors to determine if output an 'NA' in behavior classification
		self.min_length=None # the minimum length (in frames) a behavior should last, can be used to filter out the brief false positives
		self.show_legend=True # whether to show legend of behavior names in the annotated videos
		self.background_free=True # whether to include background in animations
		self.black_background=True # whether to set background black
		self.normalize_distance=True # whether to normalize the distance (in pixel) to the animal contour area
		self.social_distance=0 # a threshold (folds of size of a single animal) on whether to include individuals that are not main character in behavior examples
		self.specific_behaviors={} # sex or identity-specific behaviors
		self.correct_ID=False # whether to use sex or identity-specific behaviors to guide ID correction when ID switching is likely to happen
		#END COPY
	
		self.display_window()

	def display_window(self):

		self.panel=wx.Panel(self)
		boxsizer=wx.BoxSizer(wx.VERTICAL)

		module_inputdatafolder=wx.BoxSizer(wx.HORIZONTAL)
		self.button_inputdatafolder=wx.Button(self.panel,label='Select a folder of video clips for analysis \nto train the transformer model',size=(300,40))
		self.button_inputdatafolder.Bind(wx.EVT_BUTTON,self.select_data_input_path)
		wx.Button.SetToolTip(self.button_inputdatafolder,'Select a folder of videos for training data.')
		self.text_inputdatafolder=wx.StaticText(self.panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_inputdatafolder.Add(self.button_inputdatafolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_inputdatafolder.Add(self.text_inputdatafolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,10,0)
		boxsizer.Add(module_inputdatafolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)


		# For clip analysis
		module_selectcategorizer=wx.BoxSizer(wx.HORIZONTAL)
		self.button_selectcategorizer=wx.Button(self.panel,label='Select a Categorizer for\nbehavior classification',size=(300,40))
		self.button_selectcategorizer.Bind(wx.EVT_BUTTON,self.select_categorizer)
		wx.Button.SetToolTip(self.button_selectcategorizer,'The fps of the videos to analyze should match that of the selected Categorizer. Uncertain level determines the threshold for the Categorizer to output an ‘NA’ for behavioral classification. See Extended Guide for details.')
		self.text_selectcategorizer=wx.StaticText(self.panel,label='Default: no behavior classification, just track animals and quantify motion kinematcis.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_selectcategorizer.Add(self.button_selectcategorizer,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_selectcategorizer.Add(self.text_selectcategorizer,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,10,0)
		boxsizer.Add(module_selectcategorizer,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_detection=wx.BoxSizer(wx.HORIZONTAL)
		self.button_detection=wx.Button(self.panel,label='Specify the method to\ndetect animals or objects',size=(300,40))
		self.button_detection.Bind(wx.EVT_BUTTON,self.select_method)
		wx.Button.SetToolTip(self.button_detection,'Background subtraction-based method is accurate and fast but requires static background and stable illumination in videos; Detectors-based method is accurate and versatile in any recording settings but is slow. See Extended Guide for details.')
		self.text_detection=wx.StaticText(self.panel,label='Default: Background subtraction-based method.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_detection.Add(self.button_detection,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_detection.Add(self.text_detection,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_detection,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_selectbehaviors=wx.BoxSizer(wx.HORIZONTAL)
		self.button_selectbehaviors=wx.Button(self.panel,label='Select the behaviors for\nannotations and plots',size=(300,40))
		self.button_selectbehaviors.Bind(wx.EVT_BUTTON,self.select_behaviors)
		wx.Button.SetToolTip(self.button_selectbehaviors,'The behavior categories are determined by the selected Categorizer. Select which behaviors to show in the annotated videos / images and the raster plot (only for videos). See Extended Guide for details.')
		self.text_selectbehaviors=wx.StaticText(self.panel,label='Default: No Categorizer selected, no behavior selected.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_selectbehaviors.Add(self.button_selectbehaviors,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_selectbehaviors.Add(self.text_selectbehaviors,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_selectbehaviors,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_selectparameters=wx.BoxSizer(wx.HORIZONTAL)
		self.button_selectparameters=wx.Button(self.panel,label='Select the quantitative measurements\nfor each behavior',size=(300,40))
		self.button_selectparameters.Bind(wx.EVT_BUTTON,self.select_parameters)
		wx.Button.SetToolTip(self.button_selectparameters,'If select "not to normalize distances", all distances will be output in pixels. If select "normalize distances", all distances will be normalized to the animal size. See Extended Guide for details.')
		self.text_selectparameters=wx.StaticText(self.panel,label='Default: none.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_selectparameters.Add(self.button_selectparameters,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_selectparameters.Add(self.text_selectparameters,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_selectparameters,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)
		# end for clip analysis

		module_num_workers=wx.BoxSizer(wx.HORIZONTAL)
		self.button_num_workers=wx.Button(self.panel,label='Select the number of workers \nfor video data analysis',size=(300,40))
		self.button_num_workers.Bind(wx.EVT_BUTTON,self.select_num_workers)
		wx.Button.SetToolTip(self.button_num_workers,'Select the number of worker processes for video analysis (each worker can process one clip on one core at a time).')
		self.text_num_workers=wx.StaticText(self.panel,label="Number of workers: " + str(self.num_workers),style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_num_workers.Add(self.button_num_workers,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_num_workers.Add(self.text_num_workers,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_num_workers,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_datainput_toggle=wx.BoxSizer(wx.HORIZONTAL)
		self.toggle_input_button = wx.Button(self.panel, label='Switch to File Input for Testing', size=(200, 30))
		self.toggle_input_button.Bind(wx.EVT_BUTTON, self.select_data_input_mode)
		module_datainput_toggle.Add(self.toggle_input_button,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		self.text_toggle_input_button=wx.StaticText(self.panel,label='Option to select a folder of videos or analyzed behavior data \nfor training the transformer model.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_datainput_toggle.Add(self.text_toggle_input_button, 0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_datainput_toggle,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_outputfolder=wx.BoxSizer(wx.HORIZONTAL)
		self.button_outputfolder=wx.Button(self.panel,label='Select the folder to store the trained transformer model \nand behavior analysis data',size=(300,60))
		self.button_outputfolder.Bind(wx.EVT_BUTTON,self.select_outputpath)
		wx.Button.SetToolTip(self.button_outputfolder,'In this folder there will be a pth file containing the trained model and behavior analysis data.')
		self.text_outputfolder=wx.StaticText(self.panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_outputfolder.Add(self.button_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_outputfolder.Add(self.text_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_embedding_dims=wx.BoxSizer(wx.HORIZONTAL)
		button_embedding_dims=wx.Button(self.panel,label='Specify number of embedding dimensions\n',size=(300,40))
		button_embedding_dims.Bind(wx.EVT_BUTTON,self.input_embedding_dimensions)
		wx.Button.SetToolTip(button_embedding_dims,'Set a custom number of embedding dimensions.')
		self.text_embedding_dims=wx.StaticText(self.panel,label='Default: 128.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_embedding_dims.Add(button_embedding_dims,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_embedding_dims.Add(self.text_embedding_dims,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_embedding_dims,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_num_heads=wx.BoxSizer(wx.HORIZONTAL)
		button_num_heads=wx.Button(self.panel,label='Specify number of heads\n',size=(300,40))
		button_num_heads.Bind(wx.EVT_BUTTON,self.input_heads)
		wx.Button.SetToolTip(button_num_heads,'Set a custom number of attention heads.')
		self.text_num_heads=wx.StaticText(self.panel,label='Default: 8.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_num_heads.Add(button_num_heads,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_num_heads.Add(self.text_num_heads,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_num_heads,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_source_len=wx.BoxSizer(wx.HORIZONTAL)
		button_source_len=wx.Button(self.panel,label='Specify source length\n',size=(300,40))
		button_source_len.Bind(wx.EVT_BUTTON,self.input_source_length)
		wx.Button.SetToolTip(button_source_len,'Select how many behaviors back you would like your model to take as context.')
		self.text_source_len=wx.StaticText(self.panel,label='Default: 30.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_source_len.Add(button_source_len,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_source_len.Add(self.text_source_len,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_source_len,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_target_len=wx.BoxSizer(wx.HORIZONTAL)
		button_target_len=wx.Button(self.panel,label='Specify target length\n',size=(300,40))
		button_target_len.Bind(wx.EVT_BUTTON,self.input_target_length)
		wx.Button.SetToolTip(button_target_len,'Select how many behaviors into the future you would like to predict at a time.')
		self.text_target_len=wx.StaticText(self.panel,label='Default: 10.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_target_len.Add(button_target_len,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_target_len.Add(self.text_target_len,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_target_len,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		button_train_model=wx.Button(self.panel,label='Start to train model',size=(300,40))
		button_train_model.Bind(wx.EVT_BUTTON,self.train_model)
		wx.Button.SetToolTip(button_train_model,'A transformer model will be trained by encoding the behaviors into integer keys that can then be used to train and predict by a transformer model.')
		boxsizer.Add(0,5,0)
		boxsizer.Add(button_train_model,0,wx.RIGHT|wx.ALIGN_RIGHT,90)
		boxsizer.Add(0,10,0)

		self.panel.SetSizer(boxsizer)

		self.Centre()
		self.Show(True)

	def select_num_workers(self, event):
		dialog = wx.Dialog(self, title="Set Number of Worker Processes", size=(350, 150))

		vbox = wx.BoxSizer(wx.VERTICAL)

		num_cores = multiprocessing.cpu_count()
		st = wx.StaticText(dialog, label=f"Select number of worker processes (max {num_cores}):")
		vbox.Add(st, 0, wx.ALL | wx.EXPAND, 10)
		self.spin_workers = wx.SpinCtrl(dialog, value="", min=1, max=num_cores)
		vbox.Add(self.spin_workers, 0, wx.ALL | wx.EXPAND, 10)

		btns = dialog.CreateButtonSizer(wx.OK | wx.CANCEL)
		vbox.Add(btns, 0, wx.ALL | wx.ALIGN_CENTER, 10)

		dialog.SetSizer(vbox)

		if dialog.ShowModal() == wx.ID_OK:
			self.num_workers = self.spin_workers.GetValue()
			wx.MessageBox(f"Number of worker processes set to: {self.num_workers}", "Confirmed", wx.OK | wx.ICON_INFORMATION)
		else:
			self.num_workers = 1  # fallback default

		self.text_num_workers.SetLabel("Number of workers: " + str(self.num_workers))

		dialog.Destroy()

	def select_data_input_mode(self, event):
		if self.data_input_mode == 'video':
			self.data_input_mode = 'file'
			self.button_inputdatafolder.SetLabel('Select a folder of behavior analysis data \nto train the transformer model')
			self.button_inputdatafolder.SetToolTip('Select a folder of behavior analysis for training data.')
			self.button_outputfolder.SetLabel('Select the folder to store\nthe trained transformer model')
			self.button_outputfolder.SetToolTip('In this folder there will be a pth file containing the trained model.')
			self.toggle_input_button.SetLabel('Switch to Video Data Input')
			# Hide all children of the module_num_workers sizer
			self.button_num_workers.Hide()
			self.text_num_workers.Hide()
			self.button_selectcategorizer.Hide()
			self.text_selectcategorizer.Hide()
			self.button_detection.Hide()
			self.text_detection.Hide()
			self.button_selectbehaviors.Hide()
			self.text_selectbehaviors.Hide()
			self.button_selectparameters.Hide()
			self.text_selectparameters.Hide()
			# Reflow the layout
			self.panel.Layout()
		else:
			self.data_input_mode = 'video'
			self.button_inputdatafolder.SetLabel('Select a folder of video clips for analysis \nto train the transformer model')
			self.button_inputdatafolder.SetToolTip('Select a folder of videos for training data.')
			self.button_outputfolder.SetLabel('Select the folder to store\nthe trained transformer model and behavior analysis data')
			self.button_outputfolder.SetToolTip('In this folder there will be a pth file containing the trained model and behavior analysis data.')
			self.toggle_input_button.SetLabel('Switch to File Data Input')
			# Show all children of the module_num_workers sizer
			self.button_num_workers.Show()
			self.text_num_workers.Show()
			self.button_selectcategorizer.Show()
			self.text_selectcategorizer.Show()
			self.button_detection.Show()
			self.text_detection.Show()
			self.button_selectbehaviors.Show()
			self.text_selectbehaviors.Show()
			self.button_selectparameters.Show()
			self.text_selectparameters.Show()
			# Reflow the layout
			self.panel.Layout()
		self.text_inputdatafolder.SetLabel('None.')
		self.text_outputfolder.SetLabel('None.')


	def select_data_input_path(self,event):

		dialog=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.data_input_path=dialog.GetPath()
			if self.data_input_mode == "video":
				self.text_inputdatafolder.SetLabel('The videos to be trained off of will come from: '+self.data_input_path+'.')
			elif self.data_input_mode == "file":
				self.text_inputdatafolder.SetLabel('The behavior data to be trained off of will come from: '+self.data_input_path+'.')
		dialog.Destroy()

	def select_outputpath(self,event):

		dialog=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.out_path=dialog.GetPath()
			if self.data_input_mode == "video":
				self.text_outputfolder.SetLabel('The transformer model and behavior analysis data will be stored in: '+self.out_path+'.')
			elif self.data_input_mode == "file":
				self.text_outputfolder.SetLabel('The transformer model will be stored in: '+self.out_path+'.')
		dialog.Destroy()

	def train_model(self,event):
		if self.data_input_path is None or self.out_path is None:

			wx.MessageBox('No input / output file / folder selected.','Error',wx.OK|wx.ICON_ERROR)
			return
		
		input_folder = pathlib.Path(self.data_input_path)
		if self.data_input_mode == 'video':
			video_files = list(input_folder.glob("*.mp4"))
			output_dir = str(self.out_path)
			os.makedirs(output_dir, exist_ok=True)

			manager = multiprocessing.Manager()
			status = manager.dict()

			status["analysis_queue"] = manager.Queue()

			for clip in video_files:
				task = {"clip_filename": clip.name, "clip_folder": clip.parent}
				status["analysis_queue"].put(task)

			num_processed_clips = manager.Value("i", 0)
			num_clips = len(video_files)

			num_analysis_processes = self.num_workers
			worker_process_list = []

			for i in range(num_analysis_processes):
				worker = Train_Worker(status, self.get_train_config(), output_dir, num_processed_clips, i)
				worker_process = multiprocessing.Process(target=worker)
				worker_process_list.append(worker_process)
				worker_process.start()

			while num_processed_clips.value < num_clips:
				time.sleep(0.1)

			# Release resources
			for worker_process in worker_process_list:
				worker_process.join()
			# B/c you want to use the analyzed behavior data at the outpath
			# Otherwise, leave it b/c you already have analyzed behavior data at the in path
			self.data_input_path = self.out_path

		if self.embedding_dimensions is None:
			self.embedding_dimensions = 128  # Default value
		if self.num_heads is None:
			self.num_heads = 8  # Default value
		if self.src_len is None:
			self.src_len = 30  # Default value
		if self.tgt_len is None:
			self.tgt_len = 10  # Default value

		# Ensure embed_dim is divisible by num_heads
		if self.embedding_dimensions % self.num_heads != 0:
			self.embedding_dimensions = (self.embedding_dimensions // self.num_heads) * self.num_heads
			wx.MessageBox(
				f"Embedding dimensions adjusted to {self.embedding_dimensions} to be divisible by num_heads.",
				"Warning", wx.OK | wx.ICON_WARNING
			)

		device = "cuda" if torch.cuda.is_available() else "cpu"
		print("Using {} device for training".format(device))

		behavior_training_data = list(pathlib.Path(self.out_path).glob("*.xlsx"))
		if not len(behavior_training_data) > 0:
			print("No behavior training data")
		
		# Stack the behaviors to create the encoding
		combined_df = pd.concat(
			[pd.read_excel(xlsx_path) for xlsx_path in behavior_training_data],
			ignore_index=True
		)
		combined_path = os.path.join(self.out_path, "combined_behavior_data.xlsx")
		combined_df.to_excel(combined_path, index=False)

		master_tokenizer = LabGym_Encoder(file_path=combined_path, target_row=0)
		encoded_sequence = master_tokenizer.grab_encoded_behaviors_sequence()
		self.encodings = master_tokenizer.get_behavior_map()
		self.vocab_size = master_tokenizer.get_num_behaviors() + 1
		model = BehaviorTransformer(self.vocab_size, self.embedding_dimensions, self.num_heads, self.src_len, self.tgt_len, self.encodings).to(device)
		# Put them together to train all at once to optimize with batches
		encoded_sequence_list = []
		no_combined_behavior_training_data = [file for file in behavior_training_data if "combined_behavior_data.xlsx" not in os.path.basename(file)]
		print("No combined: " + str(no_combined_behavior_training_data))
		for xlsx_data in no_combined_behavior_training_data:
			tokenizer = LabGym_Encoder(file_path=xlsx_data, encodings=self.encodings, target_row=0)
			encoded_sequence = tokenizer.grab_encoded_behaviors_sequence()
			encoded_sequence_list.append(encoded_sequence)
			# This is some of the dumbest fucking code I've ever seen
			# Just fix your model
			# From You to You

			# trimmed_encoded_sequence = []

			# for i in range(0, len(encoded_sequence), 3):
			# 	end_idx = min(i + 3, len(encoded_sequence))
			# 	sequence = encoded_sequence[i:end_idx]

			# 	if len(sequence) > 0:
			# 		# Picks the first behavior in a tie????
			# 		trimmed_encoded_sequence.append(max(set(sequence), key=sequence.count))

			# print("Trimmed encoding sequence:", trimmed_encoded_sequence)

			# data = torch.tensor(trimmed_encoded_sequence, dtype=torch.int)

			# +1 b/c 0 is reserved for padding

			# Convert to torch tensors
			tensor_sequence_list = [torch.tensor(sequence, dtype=torch.int) for sequence in encoded_sequence_list]

		training(model, tensor_sequence_list)
		# Need to implement validation to test transformer performance next
		# For now, just blind faith lol
		# validate(model, data)
		save_model(model=model, filefolder=self.out_path)

	def input_embedding_dimensions(self, event):
		dialog = wx.MessageDialog(
			self, 
			'Would you like to set a custom embedding dimension?', 
			'Set Embedding Dimensions?', 
			wx.YES_NO | wx.ICON_QUESTION
		)
		
		if dialog.ShowModal() == wx.ID_YES:
			dialog.Destroy()
			dialog1 = wx.TextEntryDialog(self, 'Enter the embedding dimension (integer value):', 'Set Embedding Dimension')
			
			if dialog1.ShowModal() == wx.ID_OK:
				try:
					self.embedding_dimension = int(dialog1.GetValue())
					self.text_embedding_dims.SetLabel(f'Embedding dimension set to {self.embedding_dimension}.')
				except ValueError:
					wx.MessageBox('Invalid input! Please enter an integer.', 'Error', wx.OK | wx.ICON_ERROR)
					self.embedding_dimension = 128  # Reset to default
					self.text_embedding_dims.SetLabel('Invalid input. Default embedding dimension = 128.')
			
			dialog1.Destroy()
		else:
			self.embedding_dimension = 128  # Default value
			self.text_embedding_dims.SetLabel('Default embedding dimension = 128.')
		
		dialog.Destroy()

	def input_heads(self, event):
		dialog = wx.MessageDialog(
			self, 
			'Would you like to set a custom number of heads?', 
			'Set heads?', 
			wx.YES_NO | wx.ICON_QUESTION
		)
		
		if dialog.ShowModal() == wx.ID_YES:
			dialog.Destroy()
			dialog1 = wx.TextEntryDialog(self, 'Enter the number of heads (integer value):', 'Set Heads')
			
			if dialog1.ShowModal() == wx.ID_OK:
				try:
					self.heads = int(dialog1.GetValue())
					self.text_num_heads.SetLabel(f'Number of heads set to {self.heads}.')
				except ValueError:
					wx.MessageBox('Invalid input! Please enter an integer.', 'Error', wx.OK | wx.ICON_ERROR)
					self.heads = 8  # Reset to default
					self.text_num_heads.SetLabel('Invalid input. Default number of heads = 8.')
			
			dialog1.Destroy()
		else:
			self.heads = 8  # Default value
			self.text_num_heads.SetLabel('Default number of heads = 8.')
		
		dialog.Destroy()

	def input_source_length(self, event):
		dialog = wx.MessageDialog(
			self, 
			'Would you like to set a custom source length?', 
			'Set Source Length?', 
			wx.YES_NO | wx.ICON_QUESTION
		)
		
		if dialog.ShowModal() == wx.ID_YES:
			dialog.Destroy()
			dialog1 = wx.TextEntryDialog(self, 'Enter the source length (integer value):', 'Set Source Length')
			
			if dialog1.ShowModal() == wx.ID_OK:
				try:
					self.src_len = int(dialog1.GetValue())
					self.text_source_len.SetLabel(f'Source length set to {self.src_len}.')
				except ValueError:
					wx.MessageBox('Invalid input! Please enter an integer.', 'Error', wx.OK | wx.ICON_ERROR)
					self.src_len = 30  # Reset to default
					self.text_source_len.SetLabel('Invalid input. Default source length = 30.')
			
			dialog1.Destroy()
		else:
			self.src_len = 30  # Default value
			self.text_source_len.SetLabel('Default source length = 30.')
		
		dialog.Destroy()

	def input_target_length(self, event):
		dialog = wx.MessageDialog(
			self, 
			'Would you like to set a custom target length', 
			'Set Target Length?', 
			wx.YES_NO | wx.ICON_QUESTION
		)
		
		if dialog.ShowModal() == wx.ID_YES:
			dialog.Destroy()
			dialog1 = wx.TextEntryDialog(self, 'Enter the target length (integer value):', 'Set Target Length')
			
			if dialog1.ShowModal() == wx.ID_OK:
				try:
					self.tgt_len = int(dialog1.GetValue())
					self.text_target_len.SetLabel(f'Target length set to {self.tgt_len}.')
				except ValueError:
					wx.MessageBox('Invalid input! Please enter an integer.', 'Error', wx.OK | wx.ICON_ERROR)
					self.tgt_len = 10  # Reset to default
					self.text_target_len.SetLabel('Invalid input. Default target length = 10.')
			
			dialog1.Destroy()
		else:
			self.tgt_len = 10  # Default value
			self.text_target_len.SetLabel('Default target length = 10.')
		
		dialog.Destroy()
	
		# To pass LabGym analysis config to worker processes
	
	def select_method(self,event):

		if self.behavior_mode<=1:
			methods=['Subtract background (fast but requires static background & stable illumination)','Use trained Detectors (versatile but slow)']
		else:
			methods=['Use trained Detectors (versatile but slow)']

		dialog=wx.SingleChoiceDialog(self,message='How to detect the animals?',caption='Detection methods',choices=methods)

		if dialog.ShowModal()==wx.ID_OK:
			method=dialog.GetStringSelection()

			if method=='Subtract background (fast but requires static background & stable illumination)':

				self.use_detector=False

				contrasts=['Animal brighter than background','Animal darker than background','Hard to tell']
				dialog1=wx.SingleChoiceDialog(self,message='Select the scenario that fits your videos best',caption='Which fits best?',choices=contrasts)

				if dialog1.ShowModal()==wx.ID_OK:
					contrast=dialog1.GetStringSelection()
					if contrast=='Animal brighter than background':
						self.animal_vs_bg=0
					elif contrast=='Animal darker than background':
						self.animal_vs_bg=1
					else:
						self.animal_vs_bg=2
					dialog2=wx.MessageDialog(self,'Load an existing background from a folder?\nSelect "No" if dont know what it is.','(Optional) load existing background?',wx.YES_NO|wx.ICON_QUESTION)
					if dialog2.ShowModal()==wx.ID_YES:
						dialog3=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE)
						if dialog3.ShowModal()==wx.ID_OK:
							self.background_path=dialog3.GetPath()
						dialog3.Destroy()
					else:
						self.background_path=None
						if self.animal_vs_bg!=2:
							dialog3=wx.MessageDialog(self,'Unstable illumination in the video?\nSelect "Yes" if dont know what it is.','(Optional) unstable illumination?',wx.YES_NO|wx.ICON_QUESTION)
							if dialog3.ShowModal()==wx.ID_YES:
								self.stable_illumination=False
							else:
								self.stable_illumination=True
							dialog3.Destroy()
					dialog2.Destroy()

					if self.background_path is None:
						ex_methods=['Use the entire duration (default but NOT recommended)','Decode from filenames: "_xst_" and "_xet_"','Enter two time points']
						dialog2=wx.SingleChoiceDialog(self,message='Specify the time window for background extraction',caption='Time window for background extraction',choices=ex_methods)
						if dialog2.ShowModal()==wx.ID_OK:
							ex_method=dialog2.GetStringSelection()
							if ex_method=='Use the entire duration (default but NOT recommended)':
								self.decode_extraction=False
								if self.animal_vs_bg==0:
									self.text_detection.SetLabel('Background subtraction: animal brighter, using the entire duration.')
								elif self.animal_vs_bg==1:
									self.text_detection.SetLabel('Background subtraction: animal darker, using the entire duration.')
								else:
									self.text_detection.SetLabel('Background subtraction: animal partially brighter/darker, using the entire duration.')
							elif ex_method=='Decode from filenames: "_xst_" and "_xet_"':
								self.decode_extraction=True
								if self.animal_vs_bg==0:
									self.text_detection.SetLabel('Background subtraction: animal brighter, using time window decoded from filenames "_xst_" and "_xet_".')
								elif self.animal_vs_bg==1:
									self.text_detection.SetLabel('Background subtraction: animal darker, using time window decoded from filenames "_xst_" and "_xet_".')
								else:
									self.text_detection.SetLabel('Background subtraction: animal partially brighter/darker, using time window decoded from filenames "_xst_" and "_xet_".')
							else:
								self.decode_extraction=False
								dialog3=wx.NumberEntryDialog(self,'Enter the start time','The unit is second:','Start time for background extraction',0,0,100000000000000)
								if dialog3.ShowModal()==wx.ID_OK:
									self.ex_start=int(dialog3.GetValue())
								dialog3.Destroy()
								dialog3=wx.NumberEntryDialog(self,'Enter the end time','The unit is second:','End time for background extraction',0,0,100000000000000)
								if dialog3.ShowModal()==wx.ID_OK:
									self.ex_end=int(dialog3.GetValue())
									if self.ex_end==0:
										self.ex_end=None
								dialog3.Destroy()
								if self.animal_vs_bg==0:
									if self.ex_end is None:
										self.text_detection.SetLabel('Background subtraction: animal brighter, using time window (in seconds) from '+str(self.ex_start)+' to the end.')
									else:
										self.text_detection.SetLabel('Background subtraction: animal brighter, using time window (in seconds) from '+str(self.ex_start)+' to '+str(self.ex_end)+'.')
								elif self.animal_vs_bg==1:
									if self.ex_end is None:
										self.text_detection.SetLabel('Background subtraction: animal darker, using time window (in seconds) from '+str(self.ex_start)+' to the end.')
									else:
										self.text_detection.SetLabel('Background subtraction: animal darker, using time window (in seconds) from '+str(self.ex_start)+' to '+str(self.ex_end)+'.')
								else:
									if self.ex_end is None:
										self.text_detection.SetLabel('Background subtraction: animal partially brighter/darker, using time window (in seconds) from '+str(self.ex_start)+' to the end.')
									else:
										self.text_detection.SetLabel('Background subtraction: animal partially brighter/darker, using time window (in seconds) from '+str(self.ex_start)+' to '+str(self.ex_end)+'.')
						dialog2.Destroy()

				dialog1.Destroy()

			else:

				self.use_detector=True
				self.animal_number={}
				self.detector_path=os.path.join(the_absolute_current_path,'detectors')

				detectors=[i for i in os.listdir(self.detector_path) if os.path.isdir(os.path.join(self.detector_path,i))]
				if '__pycache__' in detectors:
					detectors.remove('__pycache__')
				if '__init__' in detectors:
					detectors.remove('__init__')
				if '__init__.py' in detectors:
					detectors.remove('__init__.py')
				detectors.sort()
				if 'Choose a new directory of the Detector' not in detectors:
					detectors.append('Choose a new directory of the Detector')

				dialog1=wx.SingleChoiceDialog(self,message='Select a Detector for animal detection',caption='Select a Detector',choices=detectors)
				if dialog1.ShowModal()==wx.ID_OK:
					detector=dialog1.GetStringSelection()
					if detector=='Choose a new directory of the Detector':
						dialog2=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE)
						if dialog2.ShowModal()==wx.ID_OK:
							self.path_to_detector=dialog2.GetPath()
						dialog2.Destroy()
					else:
						self.path_to_detector=os.path.join(self.detector_path,detector)
					with open(os.path.join(self.path_to_detector,'model_parameters.txt')) as f:
						model_parameters=f.read()
					animal_names=json.loads(model_parameters)['animal_names']
					if len(animal_names)>1:
						dialog2=wx.MultiChoiceDialog(self,message='Specify which animals/objects involved in analysis',caption='Animal/Object kind',choices=animal_names)
						if dialog2.ShowModal()==wx.ID_OK:
							self.animal_kinds=[animal_names[i] for i in dialog2.GetSelections()]
						else:
							self.animal_kinds=animal_names
						dialog2.Destroy()
					else:
						self.animal_kinds=animal_names
					self.animal_to_include=self.animal_kinds
					if self.behavior_mode>=3:
						dialog2=wx.NumberEntryDialog(self,"Enter the Detector's detection threshold (0~100%)","The higher detection threshold,\nthe higher detection accuracy,\nbut the lower detection sensitivity.\nEnter 0 if don't know how to set.",'Detection threshold',0,0,100)
						if dialog2.ShowModal()==wx.ID_OK:
							detection_threshold=dialog2.GetValue()
							self.detection_threshold=detection_threshold/100
						self.text_detection.SetLabel('Detector: '+detector+' (detection threshold: '+str(detection_threshold)+'%); The animals/objects: '+str(self.animal_kinds)+'.')
						dialog2.Destroy()
					else:
						for animal_name in self.animal_kinds:
							self.animal_number[animal_name]=1
						# Not needed because you don't select the number of animals, throws error in console
						# self.text_animalnumber.SetLabel('The number of '+str(self.animal_kinds)+' is: '+str(list(self.animal_number.values()))+'.')
						self.text_detection.SetLabel('Detector: '+detector+'; '+'The animals/objects: '+str(self.animal_kinds)+'.')
				dialog1.Destroy()

				if self.behavior_mode<3:
					if torch.cuda.is_available():
						dialog1=wx.NumberEntryDialog(self,'Enter the batch size for faster processing','GPU is available in this device for Detectors.\nYou may use batch processing for faster speed.','Batch size',1,1,100)
						if dialog1.ShowModal()==wx.ID_OK:
							self.detector_batch=int(dialog1.GetValue())
						else:
							self.detector_batch=1
						dialog1.Destroy()

		dialog.Destroy()

	def select_categorizer(self,event):

		if self.model_path is None:
			self.model_path=os.path.join(the_absolute_current_path,'models')

		categorizers=[i for i in os.listdir(self.model_path) if os.path.isdir(os.path.join(self.model_path,i))]
		if '__pycache__' in categorizers:
			categorizers.remove('__pycache__')
		if '__init__' in categorizers:
			categorizers.remove('__init__')
		if '__init__.py' in categorizers:
			categorizers.remove('__init__.py')
		categorizers.sort()
		if 'No behavior classification, just track animals and quantify motion kinematics' not in categorizers:
			categorizers.append('No behavior classification, just track animals and quantify motion kinematics')
		if 'Choose a new directory of the Categorizer' not in categorizers:
			categorizers.append('Choose a new directory of the Categorizer')

		dialog=wx.SingleChoiceDialog(self,message='Select a Categorizer for behavior classification',caption='Select a Categorizer',choices=categorizers)

		if dialog.ShowModal()==wx.ID_OK:
			categorizer=dialog.GetStringSelection()
			if categorizer=='Choose a new directory of the Categorizer':
				dialog1=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE)
				if dialog1.ShowModal()==wx.ID_OK:
					self.path_to_categorizer=dialog1.GetPath()
				dialog1.Destroy()
				dialog1=wx.NumberEntryDialog(self,"Enter the Categorizer's uncertainty level (0~100%)","If probability difference between\n1st- and 2nd-likely behaviors\nis less than uncertainty,\nclassfication outputs an 'NA'. Enter 0 if don't know how to set.",'Uncertainty level',0,0,100)
				if dialog1.ShowModal()==wx.ID_OK:
					uncertain=dialog1.GetValue()
					self.uncertain=uncertain/100
				else:
					uncertain=0
					self.uncertain=0
				dialog1.Destroy()
				if self.behavior_mode<3:
					dialog1=wx.MessageDialog(self,"Set a minimum length (in frames) for a behavior episode\nto output 'NA' if the duration of a identified behavior\nis shorter than the minimun length?",'Minimum length?',wx.YES_NO|wx.ICON_QUESTION)
					if dialog1.ShowModal()==wx.ID_YES:
						dialog2=wx.NumberEntryDialog(self,'Enter the minimun length (in frames)',"If the duration of a identified behavior\nis shorter than the minimun length,\nthe behavior categorization will output as 'NA'.",'Minimum length',2,1,10000)
						if dialog2.ShowModal()==wx.ID_OK:
							self.min_length=int(dialog2.GetValue())
							if self.min_length<2:
								self.min_length=2
						else:
							self.min_length=None
						dialog2.Destroy()
					else:
						self.min_length=None
					dialog1.Destroy()
				if self.min_length is None:
					self.text_selectcategorizer.SetLabel('The path to the Categorizer is: '+self.path_to_categorizer+' with uncertainty of '+str(uncertain)+'%.')
				else:
					self.text_selectcategorizer.SetLabel('The path to the Categorizer is: '+self.path_to_categorizer+' with uncertainty of '+str(uncertain)+'%; minimun length of '+str(self.min_length)+'.')
				self.text_selectbehaviors.SetLabel('All the behaviors in the selected Categorizer with default colors.')
			elif categorizer=='No behavior classification, just track animals and quantify motion kinematics':
				self.path_to_categorizer=None
				self.behavior_mode=0
				dialog1=wx.NumberEntryDialog(self,'Specify a time window used for measuring\nmotion kinematics of the tracked animals','Enter the number of\nframes (minimum=3):','Time window for calculating kinematics',15,1,100000000000000)
				if dialog1.ShowModal()==wx.ID_OK:
					self.length=int(dialog1.GetValue())
					if self.length<3:
						self.length=3
				dialog1.Destroy()
				self.text_selectcategorizer.SetLabel('No behavior classification; the time window to measure kinematics of tracked animals is: '+str(self.length)+' frames.')
				self.text_selectbehaviors.SetLabel('No behavior classification. Just track animals and quantify motion kinematics.')
			else:
				self.path_to_categorizer=os.path.join(self.model_path,categorizer)
				dialog1=wx.NumberEntryDialog(self,"Enter the Categorizer's uncertainty level (0~100%)","If probability difference between\n1st- and 2nd-likely behaviors\nis less than uncertainty,\nclassfication outputs an 'NA'.",'Uncertainty level',0,0,100)
				if dialog1.ShowModal()==wx.ID_OK:
					uncertain=dialog1.GetValue()
					self.uncertain=uncertain/100
				else:
					uncertain=0
					self.uncertain=0
				dialog1.Destroy()
				if self.behavior_mode<3:
					dialog1=wx.MessageDialog(self,"Set a minimum length (in frames) for a behavior episode\nto output 'NA' if the duration of a identified behavior\nis shorter than the minimun length?",'Minimum length?',wx.YES_NO|wx.ICON_QUESTION)
					if dialog1.ShowModal()==wx.ID_YES:
						dialog2=wx.NumberEntryDialog(self,'Enter the minimun length (in frames)',"If the duration of a identified behavior\nis shorter than the minimun length,\nthe behavior categorization will output as 'NA'.",'Minimum length',2,1,10000)
						if dialog2.ShowModal()==wx.ID_OK:
							self.min_length=int(dialog2.GetValue())
							if self.min_length<2:
								self.min_length=2
						else:
							self.min_length=None
						dialog2.Destroy()
					else:
						self.min_length=None
					dialog1.Destroy()
				if self.min_length is None:
					self.text_selectcategorizer.SetLabel('Categorizer: '+categorizer+' with uncertainty of '+str(uncertain)+'%.')
				else:
					self.text_selectcategorizer.SetLabel('Categorizer: '+categorizer+' with uncertainty of '+str(uncertain)+'%; minimun length of '+str(self.min_length)+'.')
				self.text_selectbehaviors.SetLabel('All the behaviors in the selected Categorizer with default colors.')

			if self.path_to_categorizer is not None:

				parameters=pd.read_csv(os.path.join(self.path_to_categorizer,'model_parameters.txt'))
				complete_colors=list(mpl.colors.cnames.values())
				colors=[]
				for c in complete_colors:
					colors.append(['#ffffff',c])
				self.behaviornames_and_colors={}

				for behavior_name in list(parameters['classnames']):
					index=list(parameters['classnames']).index(behavior_name)
					if index<len(colors):
						self.behaviornames_and_colors[behavior_name]=colors[index]
					else:
						self.behaviornames_and_colors[behavior_name]=['#ffffff','#ffffff']

				if 'dim_conv' in parameters:
					self.dim_conv=int(parameters['dim_conv'][0])
				if 'dim_tconv' in parameters:
					self.dim_tconv=int(parameters['dim_tconv'][0])
				self.channel=int(parameters['channel'][0])
				self.length=int(parameters['time_step'][0])
				if self.length<3:
					self.length=3
				categorizer_type=int(parameters['network'][0])
				if categorizer_type==2:
					self.animation_analyzer=True
				else:
					self.animation_analyzer=False
				if int(parameters['inner_code'][0])==0:
					self.include_bodyparts=True
				else:
					self.include_bodyparts=False
				self.std=int(parameters['std'][0])
				if int(parameters['background_free'][0])==0:
					self.background_free=True
				else:
					self.background_free=False
				if 'behavior_kind' in parameters:
					self.behavior_mode=int(parameters['behavior_kind'][0])
				else:
					self.behavior_mode=0
				if self.behavior_mode==2:
					self.social_distance=int(parameters['social_distance'][0])
					if self.social_distance==0:
						self.social_distance=float('inf')
					self.text_detection.SetLabel('Only Detector-based detection method is available for the selected Categorizer.')
				if self.behavior_mode==3:
					self.text_detection.SetLabel('Only Detector-based detection method is available for the selected Categorizer.')
					self.text_startanalyze.SetLabel('No need to specify this since the selected behavior mode is "Static images".')
					self.text_duration.SetLabel('No need to specify this since the selected behavior mode is "Static images".')
					self.text_animalnumber.SetLabel('No need to specify this since the selected behavior mode is "Static images".')
					self.text_selectparameters.SetLabel('No need to specify this since the selected behavior mode is "Static images".')
				if 'black_background' in parameters:
					if int(parameters['black_background'][0])==1:
						self.black_background=False

		dialog.Destroy()

	def select_behaviors(self,event):

		if self.path_to_categorizer is None:

			wx.MessageBox('No Categorizer selected! The behavior names are listed in the Categorizer.','Error',wx.OK|wx.ICON_ERROR)

		else:

			if len(self.animal_kinds)>1:
				dialog=wx.MultiChoiceDialog(self,message='Specify which animals/objects to annotate',caption='Animal/Object to annotate',choices=self.animal_kinds)
				if dialog.ShowModal()==wx.ID_OK:
					self.animal_to_include=[self.animal_kinds[i] for i in dialog.GetSelections()]
				else:
					self.animal_to_include=self.animal_kinds
				dialog.Destroy()
			else:
				self.animal_to_include=self.animal_kinds

			dialog=wx.MultiChoiceDialog(self,message='Select behaviors',caption='Behaviors to annotate',choices=list(self.behaviornames_and_colors.keys()))
			if dialog.ShowModal()==wx.ID_OK:
				self.behavior_to_include=[list(self.behaviornames_and_colors.keys())[i] for i in dialog.GetSelections()]
			else:
				self.behavior_to_include=list(self.behaviornames_and_colors.keys())
			dialog.Destroy()

			if len(self.behavior_to_include)==0:
				self.behavior_to_include=list(self.behaviornames_and_colors.keys())
			if self.behavior_to_include[0]=='all':
				self.behavior_to_include=list(self.behaviornames_and_colors.keys())

			if self.behavior_mode==2:
				dialog=wx.MessageDialog(self,'Specify individual-specific behaviors? e.g., sex-specific behaviors only occur in a specific sex and\ncan be used to maintain the correct ID of this individual during the entire analysis.','Individual-specific behaviors?',wx.YES_NO|wx.ICON_QUESTION)
				if dialog.ShowModal()==wx.ID_YES:
					for animal_name in self.animal_kinds:
						dialog1=wx.MultiChoiceDialog(self,message='Select individual-specific behaviors for '+str(animal_name),caption='Individual-specific behaviors for '+str(animal_name),choices=self.behavior_to_include)
						if dialog1.ShowModal()==wx.ID_OK:
							self.specific_behaviors[animal_name]={}
							self.correct_ID=True
							specific_behaviors=[self.behavior_to_include[i] for i in dialog1.GetSelections()]
							for specific_behavior in specific_behaviors:
								self.specific_behaviors[animal_name][specific_behavior]=None
						dialog1.Destroy()
				else:
					self.correct_ID=False
				dialog.Destroy()

			complete_colors=list(mpl.colors.cnames.values())
			colors=[]
			for c in complete_colors:
				colors.append(['#ffffff',c])
			
			dialog=wx.MessageDialog(self,'Specify the color to represent\nthe behaviors in annotations and plots?','Specify colors for behaviors?',wx.YES_NO|wx.ICON_QUESTION)
			if dialog.ShowModal()==wx.ID_YES:
				names_colors={}
				n=0
				while n<len(self.behavior_to_include):
					dialog2=ColorPicker(self,'Color for '+self.behavior_to_include[n],[self.behavior_to_include[n],colors[n]])
					if dialog2.ShowModal()==wx.ID_OK:
						(r,b,g,_)=dialog2.color_picker.GetColour()
						new_color='#%02x%02x%02x'%(r,b,g)
						self.behaviornames_and_colors[self.behavior_to_include[n]]=['#ffffff',new_color]
						names_colors[self.behavior_to_include[n]]=new_color
					else:
						if n<len(colors):
							names_colors[self.behavior_to_include[n]]=colors[n][1]
							self.behaviornames_and_colors[self.behavior_to_include[n]]=colors[n]
					dialog2.Destroy()
					n+=1
				if self.correct_ID:
					self.text_selectbehaviors.SetLabel('Selected: '+str(list(names_colors.keys()))+'. Specific behaviors: '+str(self.specific_behaviors)+'.')
				else:
					self.text_selectbehaviors.SetLabel('Selected: '+str(list(names_colors.keys()))+'.')
			else:
				for color in colors:
					index=colors.index(color)
					if index<len(self.behavior_to_include):
						behavior_name=list(self.behaviornames_and_colors.keys())[index]
						self.behaviornames_and_colors[behavior_name]=color
				if self.correct_ID:
					self.text_selectbehaviors.SetLabel('Selected: '+str(self.behavior_to_include)+' with default colors. Specific behaviors:'+str(self.specific_behaviors)+'.')
				else:
					self.text_selectbehaviors.SetLabel('Selected: '+str(self.behavior_to_include)+' with default colors.')
			dialog.Destroy()

			if self.behavior_mode!=3:
				dialog=wx.MessageDialog(self,'Show legend of behavior names in the annotated video?','Legend in video?',wx.YES_NO|wx.ICON_QUESTION)
				if dialog.ShowModal()==wx.ID_YES:
					self.show_legend=True
				else:
					self.show_legend=False
				dialog.Destroy()

	def select_parameters(self,event):

		if self.behavior_mode>=3:

			wx.MessageBox('No need to specify this since the selected behavior mode is "Static images".','Error',wx.OK|wx.ICON_ERROR)

		else:

			if self.path_to_categorizer is None:
				parameters=['3 areal parameters','3 length parameters','4 locomotion parameters']
			else:
				if self.behavior_mode==1:
					parameters=['count','duration','latency']
				else:
					parameters=['count','duration','latency','3 areal parameters','3 length parameters','4 locomotion parameters']

			dialog=wx.MultiChoiceDialog(self,message='Select quantitative measurements',caption='Quantitative measurements',choices=parameters)
			if dialog.ShowModal()==wx.ID_OK:
				self.parameter_to_analyze=[parameters[i] for i in dialog.GetSelections()]
			else:
				self.parameter_to_analyze=[]
			dialog.Destroy()

			if len(self.parameter_to_analyze)<=0:
				self.parameter_to_analyze=[]
				self.normalize_distance=False
				self.text_selectparameters.SetLabel('NO parameter selected.')
			else:
				if '4 locomotion parameters' in self.parameter_to_analyze:
					dialog=wx.MessageDialog(self,'Normalize the distances by the size of an animal? If no, all distances will be output in pixels.','Normalize the distances?',wx.YES_NO|wx.ICON_QUESTION)
					if dialog.ShowModal()==wx.ID_YES:
						self.normalize_distance=True
						self.text_selectparameters.SetLabel('Selected: '+str(self.parameter_to_analyze)+'; with normalization of distance.')
					else:
						self.normalize_distance=False
						self.text_selectparameters.SetLabel('Selected: '+str(self.parameter_to_analyze)+'; NO normalization of distance.')
					dialog.Destroy()
				else:
					self.normalize_distance=False
					self.text_selectparameters.SetLabel('Selected: '+str(self.parameter_to_analyze)+'.')

	def get_train_config(self):
		return {
			"out_path": self.out_path,
			"behavior_mode": self.behavior_mode,
			"use_detector": self.use_detector,
			"detector_path": self.detector_path,
			"path_to_detector": self.path_to_detector,
			"detector_batch": self.detector_batch,
			"detection_threshold": self.detection_threshold,
			"animal_kinds": self.animal_kinds,
			"background_path": self.background_path,
			"model_path": self.model_path,
			"path_to_categorizer": self.path_to_categorizer,
			"framewidth": self.framewidth,
			"delta": self.delta,
			"decode_animalnumber": self.decode_animalnumber,
			"animal_number": self.animal_number,
			"autofind_t": self.autofind_t,
			"decode_t": self.decode_t,
			"t": self.t,
			"duration": self.duration,
			"decode_extraction": self.decode_extraction,
			"ex_start": self.ex_start,
			"ex_end": self.ex_end,
			"behaviornames_and_colors": self.behaviornames_and_colors,
			"dim_tconv": self.dim_tconv,
			"dim_conv": self.dim_conv,
			"channel": self.channel,
			"length": self.length,
			"animal_vs_bg": self.animal_vs_bg,
			"stable_illumination": self.stable_illumination,
			"animation_analyzer": self.animation_analyzer,
			"animal_to_include": self.animal_to_include,
			"behavior_to_include": self.behavior_to_include,
			"parameter_to_analyze": self.parameter_to_analyze,
			"include_bodyparts": self.include_bodyparts,
			"std": self.std,
			"uncertain": self.uncertain,
			"min_length": self.min_length,
			"show_legend": self.show_legend,
			"background_free": self.background_free,
			"black_background": self.black_background,
			"normalize_distance": self.normalize_distance,
			"social_distance": self.social_distance,
			"specific_behaviors": self.specific_behaviors,
			"correct_ID": self.correct_ID,
		}

class Predict_Worker:
	def __init__(self, status, config, current_clip_number, worker_id, shutdown):
		# Dynamically assign all config keys as attributes
		for key, value in config.items():
			setattr(self, key, value)
		print("Initializing worker")
		device = "cuda" if torch.cuda.is_available() else "cpu"
		# print(f"Using {device} device")
		self.status = status
		# The most recently processed clip across all workers
		self.current_clip_number = current_clip_number
		self.shutdown = shutdown
		self.worker_id = worker_id
		print("Worker " + str(worker_id) + " initialized!")

	def append_to_master_events_spreadsheet(self, clip_results_folder):
		"""Appends new behavior analysis results to a master Excel file."""
		master_file = os.path.join(self.out_path, "all_events_master.xlsx")
		# For some reason it started saving as "all_event_probability.xlsx"?
		# new_data_file = os.path.join(clip_results_folder, "all_event_probability.xlsx")
		new_data_file = os.path.join(clip_results_folder, "all_events.xlsx")

		if not os.path.exists(master_file):
			print("Master spreadsheet not found. Creating a new one...")
			while not os.path.exists(new_data_file):
				time.sleep(0.01)
				print("Waiting for clip analysis results spreadsheet")
			shutil.copy(new_data_file, master_file)
			print(f"Created master spreadsheet: {master_file}")
			return

		while not os.path.exists(new_data_file):
			print("Waiting for clip analysis results spreadsheet")

		new_data = pd.read_excel(new_data_file)
		
		if os.path.exists(master_file):
			master_data = pd.read_excel(master_file)
			updated_data = pd.concat([master_data, new_data], ignore_index=True)
		else:
			updated_data = new_data

		updated_data.to_excel(master_file, index=False)
		print(f"Updated master spreadsheet: {master_file}")

	def append_to_master_prediction_spreadsheet(self, clip_results_folder):
		"""Appends new behavior analysis results to a master Excel file."""
		master_file = os.path.join(self.out_path, "all_predictions_master.xlsx")
		# For some reason it started saving as "all_event_probability.xlsx"?
		# new_data_file = os.path.join(clip_results_folder, "all_event_probability.xlsx")
		new_data_file = os.path.join(clip_results_folder, "all_events.xlsx")

		if not os.path.exists(master_file):
			print("Master spreadsheet not found. Creating a new one...")
			while not os.path.exists(new_data_file):
				time.sleep(0.01)
				print("Waiting for clip analysis results spreadsheet")
			shutil.copy(new_data_file, master_file)
			print(f"Created master spreadsheet: {master_file}")
			return

		while not os.path.exists(new_data_file):
			print("Waiting for clip analysis results spreadsheet")

		new_data = pd.read_excel(new_data_file)
		
		if os.path.exists(master_file):
			master_data = pd.read_excel(master_file)
			updated_data = pd.concat([master_data, new_data], ignore_index=True)
		else:
			updated_data = new_data

		updated_data.to_excel(master_file, index=False)
		print(f"Updated master spreadsheet: {master_file}")
	
	def __call__(self):
			# Was clip_analysis
			# print("Starting worker, shutdown:", self.shutdown)
			device = "cuda" if torch.cuda.is_available() else "cpu"
			self.model = load_model(filepath=self.path_to_transformer)
			self.encodings = self.model.encodings
			self.model = self.model.to(device)
			while self.shutdown.value == 0:
				time.sleep(0.1)
				# print("!!!!Queue Size: ", self.status["analysis_queue"].qsize())
				# print("!!!!")
				# print("!!!!")
				if not self.status["analysis_queue"].empty():
					print("!!!!Queue Size: ", self.status["analysis_queue"].qsize())
					print("Worker current clip number:", self.current_clip_number.value)
					task = self.status["analysis_queue"].get()
					clip_filename = task["clip_filename"]
					clip_folder = task["clip_folder"]
					print("Processing clip #", clip_filename)
					# Sleep to prevent busy waiting
					time.sleep(0.001)
					# Run behavior analysis on the captured clip
					self.path_to_videos = [clip_filename] # Set the path to the captured clip folder
					clip_results_folder = os.path.join(clip_folder, "results")
					os.makedirs(clip_results_folder, exist_ok=True)  # Create folder for results
					self.result_path = clip_results_folder  # Save output in the same directory

					# Issue is path_to_videos and result_path are self variables
					self.analyze_behaviors(event=None)

					# Make the process wait so it doesn't add behaviors out of sequence
					while task["clip_number"] > self.current_clip_number.value:
						# print("Clip number: ", task["clip_number"])
						# print("Current clip number: ", self.current_clip_number.value)
						time.sleep(.1)

					# Append results to a master spreadsheet
					self.append_to_master_events_spreadsheet(clip_results_folder)

					self.current_clip_number.value = task["clip_number"] + 1

					tokenizer = LabGym_Encoder(encodings=self.encodings, file_path=os.path.join(self.out_path, "all_events_master.xlsx"))

					encoded_sequence = tokenizer.grab_encoded_behaviors_sequence()

					# Src_len is the length of the input sequence
					if len(encoded_sequence) >= self.model.src_len:
						input_sequence = encoded_sequence[-self.model.src_len:]
						# input_sequence = [9, 9, 9, 9, 9, 9, 6, 10, 10, 10, 10, 6, 6, 6, 6, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 11, 11, 12]
						print("Input sequence:", input_sequence)
						predictions, probabilities = predict(self.model, input_sequence)
						print("Predicted next encoded behaviors:", predictions)
						predicted_behaviors = tokenizer.convert_tokens_to_behaviors(predictions)
						print("Predicted next behaviors:", predicted_behaviors)
						probs_tensor = torch.tensor(probabilities)
						pred_tensor = torch.tensor(predictions)
						# Get confidence scores for predicted behaviors
						confidences = probs_tensor.gather(-1, pred_tensor.unsqueeze(-1)).squeeze(-1).tolist()

						df = pd.DataFrame({
							"Clip Number": [task["clip_number"]] * len(predicted_behaviors),
							"Predicted Behavior": predicted_behaviors,
							"Confidence": [round(c, 4) for c in confidences]
						})

						predictions_file = os.path.join(clip_results_folder, "predictions.xlsx")
						df.to_excel(predictions_file, index=False)
						print(f"Saved predictions to {predictions_file}")
					else:
						print(f"Sequence length less than {self.model.src_len}, Skipping prediction...")

					print("Finished processing clip #", clip_filename)


	
	def analyze_behaviors(self, event):
		# Replaced all instances of self.path_to_videos and self.result_path with the path_to_videos and result_path arguments
		# Was messing up with threading because they were trying to overwrite shared instance paths

		if self.path_to_videos is None or self.result_path is None:

			wx.MessageBox('No input video(s) / result folder.','Error',wx.OK|wx.ICON_ERROR)

		else:

			if self.behavior_mode==3:

				if self.path_to_categorizer is None or self.path_to_detector is None:
					wx.MessageBox('You need to select a Categorizer / Detector.','Error',wx.OK|wx.ICON_ERROR)
				else:
					if len(self.animal_to_include)==0:
						self.animal_to_include=self.animal_kinds
					if self.detector_batch<=0:
						self.detector_batch=1
					if self.behavior_to_include[0]=='all':
						self.behavior_to_include=list(self.behaviornames_and_colors.keys())
					AAD=AnalyzeAnimalDetector()
					AAD.analyze_images_individuals(self.path_to_detector,self.path_to_videos,self.result_path,self.animal_kinds,path_to_categorizer=self.path_to_categorizer,
						generate=False,animal_to_include=self.animal_to_include,behavior_to_include=self.behavior_to_include,names_and_colors=self.behaviornames_and_colors,
						imagewidth=self.framewidth,dim_conv=self.dim_conv,channel=self.channel,detection_threshold=self.detection_threshold,uncertain=self.uncertain,
						background_free=self.background_free,black_background=self.black_background,social_distance=0)

			else:

				all_events={}
				event_data={}
				all_time=[]

				if self.use_detector:
					for animal_name in self.animal_kinds:
						all_events[animal_name]={}
					if len(self.animal_to_include)==0:
						self.animal_to_include=self.animal_kinds
					if self.detector_batch<=0:
						self.detector_batch=1

				if self.path_to_categorizer is None:
					self.behavior_to_include=[]
				else:
					if self.behavior_to_include[0]=='all':
						self.behavior_to_include=list(self.behaviornames_and_colors.keys())

				for i in self.path_to_videos:

					filename=os.path.splitext(os.path.basename(i))[0].split('_')
					if self.decode_animalnumber:
						if self.use_detector:
							self.animal_number={}
							number=[x[1:] for x in filename if len(x)>1 and x[0]=='n']
							for a,animal_name in enumerate(self.animal_kinds):
								self.animal_number[animal_name]=int(number[a])
						else:
							for x in filename:
								if len(x)>1:
									if x[0]=='n':
										self.animal_number=int(x[1:])
					if self.decode_t:
						for x in filename:
							if len(x)>1:
								if x[0]=='b':
									self.t=float(x[1:])
					if self.decode_extraction:
						for x in filename:
							if len(x)>2:
								if x[:2]=='xs':
									self.ex_start=int(x[2:])
								if x[:2]=='xe':
									self.ex_end=int(x[2:])

					if self.animal_number is None:
						if self.use_detector:
							self.animal_number={}
							for animal_name in self.animal_kinds:
								self.animal_number[animal_name]=1
						else:
							self.animal_number=1
				
					if self.path_to_categorizer is None:
						self.behavior_mode=0
						categorize_behavior=False
					else:
						categorize_behavior=True

					if self.use_detector is False:

						AA=AnalyzeAnimal()
						AA.prepare_analysis(i,self.result_path,self.animal_number,delta=self.delta,names_and_colors=self.behaviornames_and_colors,
							framewidth=self.framewidth,stable_illumination=self.stable_illumination,dim_tconv=self.dim_tconv,dim_conv=self.dim_conv,channel=self.channel,
							include_bodyparts=self.include_bodyparts,std=self.std,categorize_behavior=categorize_behavior,animation_analyzer=self.animation_analyzer,
							path_background=self.background_path,autofind_t=self.autofind_t,t=self.t,duration=self.duration,ex_start=self.ex_start,ex_end=self.ex_end,
							length=self.length,animal_vs_bg=self.animal_vs_bg)
						if self.behavior_mode==0:
							AA.acquire_information(background_free=self.background_free,black_background=self.black_background)
							AA.craft_data()
							interact_all=False
						else:
							AA.acquire_information_interact_basic(background_free=self.background_free,black_background=self.black_background)
							interact_all=True
						if self.path_to_categorizer is not None:
							AA.categorize_behaviors(self.path_to_categorizer,uncertain=self.uncertain,min_length=self.min_length)
						AA.annotate_video(self.behavior_to_include,show_legend=self.show_legend,interact_all=interact_all)
						AA.export_results(normalize_distance=self.normalize_distance,parameter_to_analyze=self.parameter_to_analyze)

						if self.path_to_categorizer is not None:
							for n in AA.event_probability:
								all_events[len(all_events)]=AA.event_probability[n]
							if len(all_time)<len(AA.all_time):
								all_time=AA.all_time

					else:

						AAD=AnalyzeAnimalDetector()
						AAD.prepare_analysis(self.path_to_detector,i,self.result_path,self.animal_number,self.animal_kinds,self.behavior_mode,
							names_and_colors=self.behaviornames_and_colors,framewidth=self.framewidth,dim_tconv=self.dim_tconv,dim_conv=self.dim_conv,channel=self.channel,
							include_bodyparts=self.include_bodyparts,std=self.std,categorize_behavior=categorize_behavior,animation_analyzer=self.animation_analyzer,
							t=self.t,duration=self.duration,length=self.length,social_distance=self.social_distance)
						if self.behavior_mode==1:
							AAD.acquire_information_interact_basic(batch_size=self.detector_batch,background_free=self.background_free,black_background=self.black_background)
						else:
							AAD.acquire_information(batch_size=self.detector_batch,background_free=self.background_free,black_background=self.black_background)
						if self.behavior_mode!=1:
							AAD.craft_data()
						if self.path_to_categorizer is not None:
							AAD.categorize_behaviors(self.path_to_categorizer,uncertain=self.uncertain,min_length=self.min_length)
						if self.correct_ID:
							AAD.correct_identity(self.specific_behaviors)
						AAD.annotate_video(self.animal_to_include,self.behavior_to_include,show_legend=self.show_legend)
						AAD.export_results(normalize_distance=self.normalize_distance,parameter_to_analyze=self.parameter_to_analyze)

						if self.path_to_categorizer is not None:
							for animal_name in self.animal_kinds:
								for n in AAD.event_probability[animal_name]:
									all_events[animal_name][len(all_events[animal_name])]=AAD.event_probability[animal_name][n]
							if len(all_time)<len(AAD.all_time):
								all_time=AAD.all_time
					
				if self.path_to_categorizer is not None:

					max_length=len(all_time)

					if self.use_detector is False:

						for n in all_events:
							event_data[len(event_data)]=all_events[n]+[['NA',-1]]*(max_length-len(all_events[n]))
						all_events_df=pd.DataFrame(event_data,index=all_time)
						all_events_df.to_excel(os.path.join(self.result_path,'all_events.xlsx'),float_format='%.2f',index_label='time/ID')
						plot_events(self.result_path,event_data,all_time,self.behaviornames_and_colors,self.behavior_to_include,width=0,height=0)
						folders=[i for i in os.listdir(self.result_path) if os.path.isdir(os.path.join(self.result_path,i))]
						folders.sort()
						for behavior_name in self.behaviornames_and_colors:
							all_summary=[]
							for folder in folders:
								individual_summary=os.path.join(self.result_path,folder,behavior_name,'all_summary.xlsx')
								if os.path.exists(individual_summary):
									all_summary.append(pd.read_excel(individual_summary))
							if len(all_summary)>=1:
								all_summary=pd.concat(all_summary,ignore_index=True)
								all_summary.to_excel(os.path.join(self.result_path,behavior_name+'_summary.xlsx'),float_format='%.2f',index_label='ID/parameter')

					else:

						for animal_name in self.animal_to_include:
							for n in all_events[animal_name]:
								event_data[len(event_data)]=all_events[animal_name][n]+[['NA',-1]]*(max_length-len(all_events[animal_name][n]))
							event_data[len(event_data)]=[['NA',-1]]*max_length
						del event_data[len(event_data)-1]
						all_events_df=pd.DataFrame(event_data,index=all_time)
						all_events_df.to_excel(os.path.join(self.result_path,'all_events.xlsx'),float_format='%.2f',index_label='time/ID')
						plot_events(self.result_path,event_data,all_time,self.behaviornames_and_colors,self.behavior_to_include,width=0,height=0)
						folders=[i for i in os.listdir(self.result_path) if os.path.isdir(os.path.join(self.result_path,i))]
						folders.sort()
						for animal_name in self.animal_kinds:
							for behavior_name in self.behaviornames_and_colors:
								all_summary=[]
								for folder in folders:
									individual_summary=os.path.join(self.result_path,folder,behavior_name,animal_name+'_all_summary.xlsx')
									if os.path.exists(individual_summary):
										all_summary.append(pd.read_excel(individual_summary))
								if len(all_summary)>=1:
									all_summary=pd.concat(all_summary,ignore_index=True)
									all_summary.to_excel(os.path.join(self.result_path,animal_name+'_'+behavior_name+'_summary.xlsx'),float_format='%.2f',index_label='ID/parameter')

			print('Analysis completed!')

class WindowLv2_LivePredictionSettings(wx.Frame):

	'''
	The 'Live Prediction Settings' window
	'''

	def __init__(self,title):

		super(WindowLv2_LivePredictionSettings,self).__init__(parent=None,title=title,size=(1000,560))
		self.path_to_transformer=None # the folder that stores LabGym analysis results
		self.out_path=None # the folder to store the transformer model
		self.analysis_interval=None # the interval between live behavior analysis in milliseconds
		self.prediction_interval=None # the number of behaviors to predict into the future

		#COPY
		self.behavior_mode=0 # 0--non-interactive, 1--interactive basic, 2--interactive advanced, 3--static images
		self.use_detector=False # whether the Detector is used
		self.detector_path=None # the 'LabGym/detectors' folder, which stores all the trained Detectors
		self.path_to_detector=None # path to the Detector
		self.detector_batch=1 # for batch processing use if GPU is available
		self.detection_threshold=0 # only for 'static images' behavior mode
		self.animal_kinds=[] # the total categories of animals / objects in a Detector
		self.background_path=None # if not None, load background images from path in 'background subtraction' detection method
		self.model_path=None # the 'LabGym/models' folder, which stores all the trained Categorizers
		self.path_to_categorizer=None # path to the Categorizer
		self.path_to_videos=None # path to a batch of videos for analysis
		self.result_path=None # the folder for storing analysis outputs
		self.framewidth=None # if not None, will resize the video frame keeping the original w:h ratio
		self.delta=10000 # the fold changes in illumination that determines the optogenetic stimulation onset
		self.decode_animalnumber=False # whether to decode animal numbers from '_nn_' in video file names
		self.animal_number=1 # the number of animals / objects in a video
		self.autofind_t=False # whether to find stimulation onset automatically (only for optogenetics)
		self.decode_t=False # whether to decode start_t from '_bt_' in video file names
		self.t=0 # the start_t for analysis
		self.duration=0 # the duration of the analysis
		self.decode_extraction=False # whether to decode time windows for background extraction from '_xst_' and '_xet_' in video file names
		self.ex_start=0 # start time for background extraction
		self.ex_end=None # end time for background extraction
		self.behaviornames_and_colors={} # behavior names in the Categorizer and their representative colors for annotation
		self.dim_tconv=8 # input dimension for Animation Analyzer in Categorizer
		self.dim_conv=8 # input dimension for Pattern Recognizer in Categorizer
		self.channel=1 # input channel for Animation Analyzer, 1--gray scale, 3--RGB scale
		self.length=15 # input time step for Animation Analyzer, also the duration / length for a behavior example
		self.animal_vs_bg=0 # 0: animals birghter than the background; 1: animals darker than the background; 2: hard to tell
		self.stable_illumination=True # whether the illumination in videos is stable
		self.animation_analyzer=True # whether to include Animation Analyzer in the Categorizers
		self.animal_to_include=[] # the animals / obejcts that will be annotated in the annotated videos / behavior plots
		self.behavior_to_include=['all'] # behaviors that will be annotated in the annotated videos / behavior plots
		self.parameter_to_analyze=[] # quantitative measures that will be included in the quantification
		self.include_bodyparts=False # whether to include body parts in the pattern images
		self.std=0 # a value between 0 and 255, higher value, less body parts will be included in the pattern images
		self.uncertain=0 # a threshold between the highest the 2nd highest probablity of behaviors to determine if output an 'NA' in behavior classification
		self.min_length=None # the minimum length (in frames) a behavior should last, can be used to filter out the brief false positives
		self.show_legend=True # whether to show legend of behavior names in the annotated videos
		self.background_free=True # whether to include background in animations
		self.black_background=True # whether to set background black
		self.normalize_distance=True # whether to normalize the distance (in pixel) to the animal contour area
		self.social_distance=0 # a threshold (folds of size of a single animal) on whether to include individuals that are not main character in behavior examples
		self.specific_behaviors={} # sex or identity-specific behaviors
		self.correct_ID=False # whether to use sex or identity-specific behaviors to guide ID correction when ID switching is likely to happen
		#END COPY
	
		self.display_window()


	def display_window(self):

		panel=wx.Panel(self)
		boxsizer=wx.BoxSizer(wx.VERTICAL)

		#COPY
		module_selectcategorizer=wx.BoxSizer(wx.HORIZONTAL)
		button_selectcategorizer=wx.Button(panel,label='Select a Categorizer for\nbehavior classification',size=(300,40))
		button_selectcategorizer.Bind(wx.EVT_BUTTON,self.select_categorizer)
		wx.Button.SetToolTip(button_selectcategorizer,'The fps of the videos to analyze should match that of the selected Categorizer. Uncertain level determines the threshold for the Categorizer to output an ‘NA’ for behavioral classification. See Extended Guide for details.')
		self.text_selectcategorizer=wx.StaticText(panel,label='Default: no behavior classification, just track animals and quantify motion kinematcis.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_selectcategorizer.Add(button_selectcategorizer,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_selectcategorizer.Add(self.text_selectcategorizer,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,10,0)
		boxsizer.Add(module_selectcategorizer,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		# REPLACE WITH CLIP

		# module_inputvideos=wx.BoxSizer(wx.HORIZONTAL)
		# button_inputvideos=wx.Button(panel,label='Select the video(s) / image(s)\nfor behavior analysis',size=(300,40))
		# button_inputvideos.Bind(wx.EVT_BUTTON,self.select_videos)
		# wx.Button.SetToolTip(button_inputvideos,'Select one or more videos / images for a behavior analysis batch. If analyzing videos, one analysis batch will yield one raster plot showing the behavior events of all the animals in all selected videos. For "Static images" mode, each annotated images will be in this folder. See Extended Guide for details.')
		# self.text_inputvideos=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		# module_inputvideos.Add(button_inputvideos,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		# module_inputvideos.Add(self.text_inputvideos,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		# boxsizer.Add(module_inputvideos,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		# boxsizer.Add(0,5,0)

		# module_outputfolder=wx.BoxSizer(wx.HORIZONTAL)
		# button_outputfolder=wx.Button(panel,label='Select a folder to store\nthe analysis results',size=(300,40))
		# button_outputfolder.Bind(wx.EVT_BUTTON,self.select_outpath)
		# wx.Button.SetToolTip(button_outputfolder,'If analyzing videos, will create a subfolder for each video in the selected folder. Each subfolder is named after the file name of the video and stores the detailed analysis results for this video. For "Static images" mode, all results will be in this folder. See Extended Guide for details.')
		# self.text_outputfolder=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		# module_outputfolder.Add(button_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		# module_outputfolder.Add(self.text_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		# boxsizer.Add(module_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		# boxsizer.Add(0,5,0)

		module_detection=wx.BoxSizer(wx.HORIZONTAL)
		button_detection=wx.Button(panel,label='Specify the method to\ndetect animals or objects',size=(300,40))
		button_detection.Bind(wx.EVT_BUTTON,self.select_method)
		wx.Button.SetToolTip(button_detection,'Background subtraction-based method is accurate and fast but requires static background and stable illumination in videos; Detectors-based method is accurate and versatile in any recording settings but is slow. See Extended Guide for details.')
		self.text_detection=wx.StaticText(panel,label='Default: Background subtraction-based method.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_detection.Add(button_detection,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_detection.Add(self.text_detection,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_detection,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		# LEAVE DEFAULT AT 0

		# module_startanalyze=wx.BoxSizer(wx.HORIZONTAL)
		# button_startanalyze=wx.Button(panel,label='Specify when the analysis\nshould begin (unit: second)',size=(300,40))
		# button_startanalyze.Bind(wx.EVT_BUTTON,self.specify_timing)
		# wx.Button.SetToolTip(button_startanalyze,'Enter a beginning time point for all videos in one analysis batch or use "Decode from filenames" to let LabGym decode the different beginning time for different videos. See Extended Guide for details.')
		# self.text_startanalyze=wx.StaticText(panel,label='Default: at the beginning of the video(s).',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		# module_startanalyze.Add(button_startanalyze,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		# module_startanalyze.Add(self.text_startanalyze,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		# boxsizer.Add(module_startanalyze,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		# boxsizer.Add(0,5,0)

		# LEAVE DEFAULT TILL END

		# module_duration=wx.BoxSizer(wx.HORIZONTAL)
		# button_duration=wx.Button(panel,label='Specify the analysis duration\n(unit: second)',size=(300,40))
		# button_duration.Bind(wx.EVT_BUTTON,self.input_duration)
		# wx.Button.SetToolTip(button_duration,'The duration is the same for all the videos in a same analysis batch.')
		# self.text_duration=wx.StaticText(panel,label='Default: from the specified beginning time to the end of a video.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		# module_duration.Add(button_duration,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		# module_duration.Add(self.text_duration,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		# boxsizer.Add(module_duration,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		# boxsizer.Add(0,5,0)

		# ONLY WORKS FOR ONE ANIMAL FOR NOW SO LEAVE DEFAULT AT 1

		# module_animalnumber=wx.BoxSizer(wx.HORIZONTAL)
		# button_animalnumber=wx.Button(panel,label='Specify the number of animals\nin a video',size=(300,40))
		# button_animalnumber.Bind(wx.EVT_BUTTON,self.specify_animalnumber)
		# wx.Button.SetToolTip(button_animalnumber,'Enter a number for all videos in one analysis batch or use "Decode from filenames" to let LabGym decode the different animal number for different videos. See Extended Guide for details.')
		# self.text_animalnumber=wx.StaticText(panel,label='Default: 1.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		# module_animalnumber.Add(button_animalnumber,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		# module_animalnumber.Add(self.text_animalnumber,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		# boxsizer.Add(module_animalnumber,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		# boxsizer.Add(0,5,0)

		module_selectbehaviors=wx.BoxSizer(wx.HORIZONTAL)
		button_selectbehaviors=wx.Button(panel,label='Select the behaviors for\nannotations and plots',size=(300,40))
		button_selectbehaviors.Bind(wx.EVT_BUTTON,self.select_behaviors)
		wx.Button.SetToolTip(button_selectbehaviors,'The behavior categories are determined by the selected Categorizer. Select which behaviors to show in the annotated videos / images and the raster plot (only for videos). See Extended Guide for details.')
		self.text_selectbehaviors=wx.StaticText(panel,label='Default: No Categorizer selected, no behavior selected.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_selectbehaviors.Add(button_selectbehaviors,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_selectbehaviors.Add(self.text_selectbehaviors,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_selectbehaviors,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_selectparameters=wx.BoxSizer(wx.HORIZONTAL)
		button_selectparameters=wx.Button(panel,label='Select the quantitative measurements\nfor each behavior',size=(300,40))
		button_selectparameters.Bind(wx.EVT_BUTTON,self.select_parameters)
		wx.Button.SetToolTip(button_selectparameters,'If select "not to normalize distances", all distances will be output in pixels. If select "normalize distances", all distances will be normalized to the animal size. See Extended Guide for details.')
		self.text_selectparameters=wx.StaticText(panel,label='Default: none.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_selectparameters.Add(button_selectparameters,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_selectparameters.Add(self.text_selectparameters,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_selectparameters,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		#END COPY

		module_inputfile=wx.BoxSizer(wx.HORIZONTAL)
		button_inputfile=wx.Button(panel,label='Select the file that stores\nthe prediction transformer model',size=(300,40))
		button_inputfile.Bind(wx.EVT_BUTTON,self.select_inputpath)
		wx.Button.SetToolTip(button_inputfile,'This is the file that stores the trained transformer model')
		self.text_inputfile=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_inputfile.Add(button_inputfile,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_inputfile.Add(self.text_inputfile,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,10,0)
		boxsizer.Add(module_inputfile,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_outputfolder=wx.BoxSizer(wx.HORIZONTAL)
		button_outputfolder=wx.Button(panel,label='Select the folder to store\nthe analyzed and predicted behaviors',size=(300,40))
		button_outputfolder.Bind(wx.EVT_BUTTON,self.select_outputpath)
		wx.Button.SetToolTip(button_outputfolder,'In this folder there will be a file containing the live analysis and prediction output.')
		self.text_outputfolder=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_outputfolder.Add(button_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_outputfolder.Add(self.text_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_analysis_interval=wx.BoxSizer(wx.HORIZONTAL)
		button_analysis_interval=wx.Button(panel,label='Specify the interval of capture (in milliseconds)\nfor live behavior analysis',size=(300,40))
		button_analysis_interval.Bind(wx.EVT_BUTTON,self.input_analysis_interval)
		wx.Button.SetToolTip(button_analysis_interval,'Set a custom number of milliseconds for the clip length for each behavior analysis.')
		self.text_analysis_interval=wx.StaticText(panel,label='Default: 5000 (5 seconds).',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_analysis_interval.Add(button_analysis_interval,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_analysis_interval.Add(self.text_analysis_interval,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_analysis_interval,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		# module_prediction_interval=wx.BoxSizer(wx.HORIZONTAL)
		# button_prediction_interval=wx.Button(panel,label='Specify how many behavior analysis into\nthe future you would like to predict',size=(300,40))
		# button_prediction_interval.Bind(wx.EVT_BUTTON,self.input_prediction_interval)
		# wx.Button.SetToolTip(button_prediction_interval,'Set how many behaviors to predict based off of the interval above.')
		# self.text_prediction_interval=wx.StaticText(panel,label='Default: 4.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		# module_prediction_interval.Add(button_prediction_interval,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		# module_prediction_interval.Add(self.text_prediction_interval,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		# boxsizer.Add(module_prediction_interval,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		# boxsizer.Add(0,5,0)

		button_start_analysis=wx.Button(panel,label='Start analysis',size=(300,40))
		button_start_analysis.Bind(wx.EVT_BUTTON,self.start_analysis)
		wx.Button.SetToolTip(button_start_analysis,'Start live analysis.')
		boxsizer.Add(0,5,0)
		boxsizer.Add(button_start_analysis,0,wx.RIGHT|wx.ALIGN_RIGHT,90)
		boxsizer.Add(0,10,0)

		panel.SetSizer(boxsizer)

		self.Centre()
		self.Show(True)

	#COPY

	def select_method(self,event):

		if self.behavior_mode<=1:
			methods=['Subtract background (fast but requires static background & stable illumination)','Use trained Detectors (versatile but slow)']
		else:
			methods=['Use trained Detectors (versatile but slow)']

		dialog=wx.SingleChoiceDialog(self,message='How to detect the animals?',caption='Detection methods',choices=methods)

		if dialog.ShowModal()==wx.ID_OK:
			method=dialog.GetStringSelection()

			if method=='Subtract background (fast but requires static background & stable illumination)':

				self.use_detector=False

				contrasts=['Animal brighter than background','Animal darker than background','Hard to tell']
				dialog1=wx.SingleChoiceDialog(self,message='Select the scenario that fits your videos best',caption='Which fits best?',choices=contrasts)

				if dialog1.ShowModal()==wx.ID_OK:
					contrast=dialog1.GetStringSelection()
					if contrast=='Animal brighter than background':
						self.animal_vs_bg=0
					elif contrast=='Animal darker than background':
						self.animal_vs_bg=1
					else:
						self.animal_vs_bg=2
					dialog2=wx.MessageDialog(self,'Load an existing background from a folder?\nSelect "No" if dont know what it is.','(Optional) load existing background?',wx.YES_NO|wx.ICON_QUESTION)
					if dialog2.ShowModal()==wx.ID_YES:
						dialog3=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE)
						if dialog3.ShowModal()==wx.ID_OK:
							self.background_path=dialog3.GetPath()
						dialog3.Destroy()
					else:
						self.background_path=None
						if self.animal_vs_bg!=2:
							dialog3=wx.MessageDialog(self,'Unstable illumination in the video?\nSelect "Yes" if dont know what it is.','(Optional) unstable illumination?',wx.YES_NO|wx.ICON_QUESTION)
							if dialog3.ShowModal()==wx.ID_YES:
								self.stable_illumination=False
							else:
								self.stable_illumination=True
							dialog3.Destroy()
					dialog2.Destroy()

					if self.background_path is None:
						ex_methods=['Use the entire duration (default but NOT recommended)','Decode from filenames: "_xst_" and "_xet_"','Enter two time points']
						dialog2=wx.SingleChoiceDialog(self,message='Specify the time window for background extraction',caption='Time window for background extraction',choices=ex_methods)
						if dialog2.ShowModal()==wx.ID_OK:
							ex_method=dialog2.GetStringSelection()
							if ex_method=='Use the entire duration (default but NOT recommended)':
								self.decode_extraction=False
								if self.animal_vs_bg==0:
									self.text_detection.SetLabel('Background subtraction: animal brighter, using the entire duration.')
								elif self.animal_vs_bg==1:
									self.text_detection.SetLabel('Background subtraction: animal darker, using the entire duration.')
								else:
									self.text_detection.SetLabel('Background subtraction: animal partially brighter/darker, using the entire duration.')
							elif ex_method=='Decode from filenames: "_xst_" and "_xet_"':
								self.decode_extraction=True
								if self.animal_vs_bg==0:
									self.text_detection.SetLabel('Background subtraction: animal brighter, using time window decoded from filenames "_xst_" and "_xet_".')
								elif self.animal_vs_bg==1:
									self.text_detection.SetLabel('Background subtraction: animal darker, using time window decoded from filenames "_xst_" and "_xet_".')
								else:
									self.text_detection.SetLabel('Background subtraction: animal partially brighter/darker, using time window decoded from filenames "_xst_" and "_xet_".')
							else:
								self.decode_extraction=False
								dialog3=wx.NumberEntryDialog(self,'Enter the start time','The unit is second:','Start time for background extraction',0,0,100000000000000)
								if dialog3.ShowModal()==wx.ID_OK:
									self.ex_start=int(dialog3.GetValue())
								dialog3.Destroy()
								dialog3=wx.NumberEntryDialog(self,'Enter the end time','The unit is second:','End time for background extraction',0,0,100000000000000)
								if dialog3.ShowModal()==wx.ID_OK:
									self.ex_end=int(dialog3.GetValue())
									if self.ex_end==0:
										self.ex_end=None
								dialog3.Destroy()
								if self.animal_vs_bg==0:
									if self.ex_end is None:
										self.text_detection.SetLabel('Background subtraction: animal brighter, using time window (in seconds) from '+str(self.ex_start)+' to the end.')
									else:
										self.text_detection.SetLabel('Background subtraction: animal brighter, using time window (in seconds) from '+str(self.ex_start)+' to '+str(self.ex_end)+'.')
								elif self.animal_vs_bg==1:
									if self.ex_end is None:
										self.text_detection.SetLabel('Background subtraction: animal darker, using time window (in seconds) from '+str(self.ex_start)+' to the end.')
									else:
										self.text_detection.SetLabel('Background subtraction: animal darker, using time window (in seconds) from '+str(self.ex_start)+' to '+str(self.ex_end)+'.')
								else:
									if self.ex_end is None:
										self.text_detection.SetLabel('Background subtraction: animal partially brighter/darker, using time window (in seconds) from '+str(self.ex_start)+' to the end.')
									else:
										self.text_detection.SetLabel('Background subtraction: animal partially brighter/darker, using time window (in seconds) from '+str(self.ex_start)+' to '+str(self.ex_end)+'.')
						dialog2.Destroy()

				dialog1.Destroy()

			else:

				self.use_detector=True
				self.animal_number={}
				self.detector_path=os.path.join(the_absolute_current_path,'detectors')

				detectors=[i for i in os.listdir(self.detector_path) if os.path.isdir(os.path.join(self.detector_path,i))]
				if '__pycache__' in detectors:
					detectors.remove('__pycache__')
				if '__init__' in detectors:
					detectors.remove('__init__')
				if '__init__.py' in detectors:
					detectors.remove('__init__.py')
				detectors.sort()
				if 'Choose a new directory of the Detector' not in detectors:
					detectors.append('Choose a new directory of the Detector')

				dialog1=wx.SingleChoiceDialog(self,message='Select a Detector for animal detection',caption='Select a Detector',choices=detectors)
				if dialog1.ShowModal()==wx.ID_OK:
					detector=dialog1.GetStringSelection()
					if detector=='Choose a new directory of the Detector':
						dialog2=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE)
						if dialog2.ShowModal()==wx.ID_OK:
							self.path_to_detector=dialog2.GetPath()
						dialog2.Destroy()
					else:
						self.path_to_detector=os.path.join(self.detector_path,detector)
					with open(os.path.join(self.path_to_detector,'model_parameters.txt')) as f:
						model_parameters=f.read()
					animal_names=json.loads(model_parameters)['animal_names']
					if len(animal_names)>1:
						dialog2=wx.MultiChoiceDialog(self,message='Specify which animals/objects involved in analysis',caption='Animal/Object kind',choices=animal_names)
						if dialog2.ShowModal()==wx.ID_OK:
							self.animal_kinds=[animal_names[i] for i in dialog2.GetSelections()]
						else:
							self.animal_kinds=animal_names
						dialog2.Destroy()
					else:
						self.animal_kinds=animal_names
					self.animal_to_include=self.animal_kinds
					if self.behavior_mode>=3:
						dialog2=wx.NumberEntryDialog(self,"Enter the Detector's detection threshold (0~100%)","The higher detection threshold,\nthe higher detection accuracy,\nbut the lower detection sensitivity.\nEnter 0 if don't know how to set.",'Detection threshold',0,0,100)
						if dialog2.ShowModal()==wx.ID_OK:
							detection_threshold=dialog2.GetValue()
							self.detection_threshold=detection_threshold/100
						self.text_detection.SetLabel('Detector: '+detector+' (detection threshold: '+str(detection_threshold)+'%); The animals/objects: '+str(self.animal_kinds)+'.')
						dialog2.Destroy()
					else:
						for animal_name in self.animal_kinds:
							self.animal_number[animal_name]=1
						# Not needed because you don't select the number of animals, throws error in console
						# self.text_animalnumber.SetLabel('The number of '+str(self.animal_kinds)+' is: '+str(list(self.animal_number.values()))+'.')
						self.text_detection.SetLabel('Detector: '+detector+'; '+'The animals/objects: '+str(self.animal_kinds)+'.')
				dialog1.Destroy()

				if self.behavior_mode<3:
					if torch.cuda.is_available():
						dialog1=wx.NumberEntryDialog(self,'Enter the batch size for faster processing','GPU is available in this device for Detectors.\nYou may use batch processing for faster speed.','Batch size',1,1,100)
						if dialog1.ShowModal()==wx.ID_OK:
							self.detector_batch=int(dialog1.GetValue())
						else:
							self.detector_batch=1
						dialog1.Destroy()

		dialog.Destroy()

	def select_categorizer(self,event):

		if self.model_path is None:
			self.model_path=os.path.join(the_absolute_current_path,'models')

		categorizers=[i for i in os.listdir(self.model_path) if os.path.isdir(os.path.join(self.model_path,i))]
		if '__pycache__' in categorizers:
			categorizers.remove('__pycache__')
		if '__init__' in categorizers:
			categorizers.remove('__init__')
		if '__init__.py' in categorizers:
			categorizers.remove('__init__.py')
		categorizers.sort()
		if 'No behavior classification, just track animals and quantify motion kinematics' not in categorizers:
			categorizers.append('No behavior classification, just track animals and quantify motion kinematics')
		if 'Choose a new directory of the Categorizer' not in categorizers:
			categorizers.append('Choose a new directory of the Categorizer')

		dialog=wx.SingleChoiceDialog(self,message='Select a Categorizer for behavior classification',caption='Select a Categorizer',choices=categorizers)

		if dialog.ShowModal()==wx.ID_OK:
			categorizer=dialog.GetStringSelection()
			if categorizer=='Choose a new directory of the Categorizer':
				dialog1=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE)
				if dialog1.ShowModal()==wx.ID_OK:
					self.path_to_categorizer=dialog1.GetPath()
				dialog1.Destroy()
				dialog1=wx.NumberEntryDialog(self,"Enter the Categorizer's uncertainty level (0~100%)","If probability difference between\n1st- and 2nd-likely behaviors\nis less than uncertainty,\nclassfication outputs an 'NA'. Enter 0 if don't know how to set.",'Uncertainty level',0,0,100)
				if dialog1.ShowModal()==wx.ID_OK:
					uncertain=dialog1.GetValue()
					self.uncertain=uncertain/100
				else:
					uncertain=0
					self.uncertain=0
				dialog1.Destroy()
				if self.behavior_mode<3:
					dialog1=wx.MessageDialog(self,"Set a minimum length (in frames) for a behavior episode\nto output 'NA' if the duration of a identified behavior\nis shorter than the minimun length?",'Minimum length?',wx.YES_NO|wx.ICON_QUESTION)
					if dialog1.ShowModal()==wx.ID_YES:
						dialog2=wx.NumberEntryDialog(self,'Enter the minimun length (in frames)',"If the duration of a identified behavior\nis shorter than the minimun length,\nthe behavior categorization will output as 'NA'.",'Minimum length',2,1,10000)
						if dialog2.ShowModal()==wx.ID_OK:
							self.min_length=int(dialog2.GetValue())
							if self.min_length<2:
								self.min_length=2
						else:
							self.min_length=None
						dialog2.Destroy()
					else:
						self.min_length=None
					dialog1.Destroy()
				if self.min_length is None:
					self.text_selectcategorizer.SetLabel('The path to the Categorizer is: '+self.path_to_categorizer+' with uncertainty of '+str(uncertain)+'%.')
				else:
					self.text_selectcategorizer.SetLabel('The path to the Categorizer is: '+self.path_to_categorizer+' with uncertainty of '+str(uncertain)+'%; minimun length of '+str(self.min_length)+'.')
				self.text_selectbehaviors.SetLabel('All the behaviors in the selected Categorizer with default colors.')
			elif categorizer=='No behavior classification, just track animals and quantify motion kinematics':
				self.path_to_categorizer=None
				self.behavior_mode=0
				dialog1=wx.NumberEntryDialog(self,'Specify a time window used for measuring\nmotion kinematics of the tracked animals','Enter the number of\nframes (minimum=3):','Time window for calculating kinematics',15,1,100000000000000)
				if dialog1.ShowModal()==wx.ID_OK:
					self.length=int(dialog1.GetValue())
					if self.length<3:
						self.length=3
				dialog1.Destroy()
				self.text_selectcategorizer.SetLabel('No behavior classification; the time window to measure kinematics of tracked animals is: '+str(self.length)+' frames.')
				self.text_selectbehaviors.SetLabel('No behavior classification. Just track animals and quantify motion kinematics.')
			else:
				self.path_to_categorizer=os.path.join(self.model_path,categorizer)
				dialog1=wx.NumberEntryDialog(self,"Enter the Categorizer's uncertainty level (0~100%)","If probability difference between\n1st- and 2nd-likely behaviors\nis less than uncertainty,\nclassfication outputs an 'NA'.",'Uncertainty level',0,0,100)
				if dialog1.ShowModal()==wx.ID_OK:
					uncertain=dialog1.GetValue()
					self.uncertain=uncertain/100
				else:
					uncertain=0
					self.uncertain=0
				dialog1.Destroy()
				if self.behavior_mode<3:
					dialog1=wx.MessageDialog(self,"Set a minimum length (in frames) for a behavior episode\nto output 'NA' if the duration of a identified behavior\nis shorter than the minimun length?",'Minimum length?',wx.YES_NO|wx.ICON_QUESTION)
					if dialog1.ShowModal()==wx.ID_YES:
						dialog2=wx.NumberEntryDialog(self,'Enter the minimun length (in frames)',"If the duration of a identified behavior\nis shorter than the minimun length,\nthe behavior categorization will output as 'NA'.",'Minimum length',2,1,10000)
						if dialog2.ShowModal()==wx.ID_OK:
							self.min_length=int(dialog2.GetValue())
							if self.min_length<2:
								self.min_length=2
						else:
							self.min_length=None
						dialog2.Destroy()
					else:
						self.min_length=None
					dialog1.Destroy()
				if self.min_length is None:
					self.text_selectcategorizer.SetLabel('Categorizer: '+categorizer+' with uncertainty of '+str(uncertain)+'%.')
				else:
					self.text_selectcategorizer.SetLabel('Categorizer: '+categorizer+' with uncertainty of '+str(uncertain)+'%; minimun length of '+str(self.min_length)+'.')
				self.text_selectbehaviors.SetLabel('All the behaviors in the selected Categorizer with default colors.')

			if self.path_to_categorizer is not None:

				parameters=pd.read_csv(os.path.join(self.path_to_categorizer,'model_parameters.txt'))
				complete_colors=list(mpl.colors.cnames.values())
				colors=[]
				for c in complete_colors:
					colors.append(['#ffffff',c])
				self.behaviornames_and_colors={}

				for behavior_name in list(parameters['classnames']):
					index=list(parameters['classnames']).index(behavior_name)
					if index<len(colors):
						self.behaviornames_and_colors[behavior_name]=colors[index]
					else:
						self.behaviornames_and_colors[behavior_name]=['#ffffff','#ffffff']

				if 'dim_conv' in parameters:
					self.dim_conv=int(parameters['dim_conv'][0])
				if 'dim_tconv' in parameters:
					self.dim_tconv=int(parameters['dim_tconv'][0])
				self.channel=int(parameters['channel'][0])
				self.length=int(parameters['time_step'][0])
				if self.length<3:
					self.length=3
				categorizer_type=int(parameters['network'][0])
				if categorizer_type==2:
					self.animation_analyzer=True
				else:
					self.animation_analyzer=False
				if int(parameters['inner_code'][0])==0:
					self.include_bodyparts=True
				else:
					self.include_bodyparts=False
				self.std=int(parameters['std'][0])
				if int(parameters['background_free'][0])==0:
					self.background_free=True
				else:
					self.background_free=False
				if 'behavior_kind' in parameters:
					self.behavior_mode=int(parameters['behavior_kind'][0])
				else:
					self.behavior_mode=0
				if self.behavior_mode==2:
					self.social_distance=int(parameters['social_distance'][0])
					if self.social_distance==0:
						self.social_distance=float('inf')
					self.text_detection.SetLabel('Only Detector-based detection method is available for the selected Categorizer.')
				if self.behavior_mode==3:
					self.text_detection.SetLabel('Only Detector-based detection method is available for the selected Categorizer.')
					self.text_startanalyze.SetLabel('No need to specify this since the selected behavior mode is "Static images".')
					self.text_duration.SetLabel('No need to specify this since the selected behavior mode is "Static images".')
					self.text_animalnumber.SetLabel('No need to specify this since the selected behavior mode is "Static images".')
					self.text_selectparameters.SetLabel('No need to specify this since the selected behavior mode is "Static images".')
				if 'black_background' in parameters:
					if int(parameters['black_background'][0])==1:
						self.black_background=False

		dialog.Destroy()

	def select_behaviors(self,event):

		if self.path_to_categorizer is None:

			wx.MessageBox('No Categorizer selected! The behavior names are listed in the Categorizer.','Error',wx.OK|wx.ICON_ERROR)

		else:

			if len(self.animal_kinds)>1:
				dialog=wx.MultiChoiceDialog(self,message='Specify which animals/objects to annotate',caption='Animal/Object to annotate',choices=self.animal_kinds)
				if dialog.ShowModal()==wx.ID_OK:
					self.animal_to_include=[self.animal_kinds[i] for i in dialog.GetSelections()]
				else:
					self.animal_to_include=self.animal_kinds
				dialog.Destroy()
			else:
				self.animal_to_include=self.animal_kinds

			dialog=wx.MultiChoiceDialog(self,message='Select behaviors',caption='Behaviors to annotate',choices=list(self.behaviornames_and_colors.keys()))
			if dialog.ShowModal()==wx.ID_OK:
				self.behavior_to_include=[list(self.behaviornames_and_colors.keys())[i] for i in dialog.GetSelections()]
			else:
				self.behavior_to_include=list(self.behaviornames_and_colors.keys())
			dialog.Destroy()

			if len(self.behavior_to_include)==0:
				self.behavior_to_include=list(self.behaviornames_and_colors.keys())
			if self.behavior_to_include[0]=='all':
				self.behavior_to_include=list(self.behaviornames_and_colors.keys())

			if self.behavior_mode==2:
				dialog=wx.MessageDialog(self,'Specify individual-specific behaviors? e.g., sex-specific behaviors only occur in a specific sex and\ncan be used to maintain the correct ID of this individual during the entire analysis.','Individual-specific behaviors?',wx.YES_NO|wx.ICON_QUESTION)
				if dialog.ShowModal()==wx.ID_YES:
					for animal_name in self.animal_kinds:
						dialog1=wx.MultiChoiceDialog(self,message='Select individual-specific behaviors for '+str(animal_name),caption='Individual-specific behaviors for '+str(animal_name),choices=self.behavior_to_include)
						if dialog1.ShowModal()==wx.ID_OK:
							self.specific_behaviors[animal_name]={}
							self.correct_ID=True
							specific_behaviors=[self.behavior_to_include[i] for i in dialog1.GetSelections()]
							for specific_behavior in specific_behaviors:
								self.specific_behaviors[animal_name][specific_behavior]=None
						dialog1.Destroy()
				else:
					self.correct_ID=False
				dialog.Destroy()

			complete_colors=list(mpl.colors.cnames.values())
			colors=[]
			for c in complete_colors:
				colors.append(['#ffffff',c])
			
			dialog=wx.MessageDialog(self,'Specify the color to represent\nthe behaviors in annotations and plots?','Specify colors for behaviors?',wx.YES_NO|wx.ICON_QUESTION)
			if dialog.ShowModal()==wx.ID_YES:
				names_colors={}
				n=0
				while n<len(self.behavior_to_include):
					dialog2=ColorPicker(self,'Color for '+self.behavior_to_include[n],[self.behavior_to_include[n],colors[n]])
					if dialog2.ShowModal()==wx.ID_OK:
						(r,b,g,_)=dialog2.color_picker.GetColour()
						new_color='#%02x%02x%02x'%(r,b,g)
						self.behaviornames_and_colors[self.behavior_to_include[n]]=['#ffffff',new_color]
						names_colors[self.behavior_to_include[n]]=new_color
					else:
						if n<len(colors):
							names_colors[self.behavior_to_include[n]]=colors[n][1]
							self.behaviornames_and_colors[self.behavior_to_include[n]]=colors[n]
					dialog2.Destroy()
					n+=1
				if self.correct_ID:
					self.text_selectbehaviors.SetLabel('Selected: '+str(list(names_colors.keys()))+'. Specific behaviors: '+str(self.specific_behaviors)+'.')
				else:
					self.text_selectbehaviors.SetLabel('Selected: '+str(list(names_colors.keys()))+'.')
			else:
				for color in colors:
					index=colors.index(color)
					if index<len(self.behavior_to_include):
						behavior_name=list(self.behaviornames_and_colors.keys())[index]
						self.behaviornames_and_colors[behavior_name]=color
				if self.correct_ID:
					self.text_selectbehaviors.SetLabel('Selected: '+str(self.behavior_to_include)+' with default colors. Specific behaviors:'+str(self.specific_behaviors)+'.')
				else:
					self.text_selectbehaviors.SetLabel('Selected: '+str(self.behavior_to_include)+' with default colors.')
			dialog.Destroy()

			if self.behavior_mode!=3:
				dialog=wx.MessageDialog(self,'Show legend of behavior names in the annotated video?','Legend in video?',wx.YES_NO|wx.ICON_QUESTION)
				if dialog.ShowModal()==wx.ID_YES:
					self.show_legend=True
				else:
					self.show_legend=False
				dialog.Destroy()


	def select_parameters(self,event):

		if self.behavior_mode>=3:

			wx.MessageBox('No need to specify this since the selected behavior mode is "Static images".','Error',wx.OK|wx.ICON_ERROR)

		else:

			if self.path_to_categorizer is None:
				parameters=['3 areal parameters','3 length parameters','4 locomotion parameters']
			else:
				if self.behavior_mode==1:
					parameters=['count','duration','latency']
				else:
					parameters=['count','duration','latency','3 areal parameters','3 length parameters','4 locomotion parameters']

			dialog=wx.MultiChoiceDialog(self,message='Select quantitative measurements',caption='Quantitative measurements',choices=parameters)
			if dialog.ShowModal()==wx.ID_OK:
				self.parameter_to_analyze=[parameters[i] for i in dialog.GetSelections()]
			else:
				self.parameter_to_analyze=[]
			dialog.Destroy()

			if len(self.parameter_to_analyze)<=0:
				self.parameter_to_analyze=[]
				self.normalize_distance=False
				self.text_selectparameters.SetLabel('NO parameter selected.')
			else:
				if '4 locomotion parameters' in self.parameter_to_analyze:
					dialog=wx.MessageDialog(self,'Normalize the distances by the size of an animal? If no, all distances will be output in pixels.','Normalize the distances?',wx.YES_NO|wx.ICON_QUESTION)
					if dialog.ShowModal()==wx.ID_YES:
						self.normalize_distance=True
						self.text_selectparameters.SetLabel('Selected: '+str(self.parameter_to_analyze)+'; with normalization of distance.')
					else:
						self.normalize_distance=False
						self.text_selectparameters.SetLabel('Selected: '+str(self.parameter_to_analyze)+'; NO normalization of distance.')
					dialog.Destroy()
				else:
					self.normalize_distance=False
					self.text_selectparameters.SetLabel('Selected: '+str(self.parameter_to_analyze)+'.')

	#END COPY


	def select_inputpath(self,event):

		dialog=wx.FileDialog(self,'Select the file storing the transformer model: ','',style=wx.DD_DEFAULT_STYLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.path_to_transformer=dialog.GetPath()
			self.text_inputfile.SetLabel('Selected: '+self.path_to_transformer+'.')
		dialog.Destroy()


	def select_outputpath(self,event):

		dialog=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.out_path=dialog.GetPath()
			self.result_path = self.out_path
			self.text_outputfolder.SetLabel('The output will be in: '+self.out_path+'.')
		dialog.Destroy()

	def input_analysis_interval(self, event):
		dialog = wx.MessageDialog(
			self, 
			'Would you like to set a custom analysis interval?', 
			'Set Analysis Interval?', 
			wx.YES_NO | wx.ICON_QUESTION
		)
		
		if dialog.ShowModal() == wx.ID_YES:
			dialog.Destroy()
			dialog1 = wx.TextEntryDialog(self, 'Enter the custom analysis interval (integer representing milliseconds):', 'Set Analysis Interval')
			
			if dialog1.ShowModal() == wx.ID_OK:
				try:
					self.analysis_interval = int(dialog1.GetValue())
					self.text_analysis_interval.SetLabel(f'Anlysis interval set to {self.analysis_interval}.')
				except ValueError:
					wx.MessageBox('Invalid input! Please enter an integer.', 'Error', wx.OK | wx.ICON_ERROR)
					self.analysis_interval = 5000  # Reset to default
					self.text_analysis_interval.SetLabel('Invalid input. Default analysis interval = 5000 (milliseconds).')
			
			dialog1.Destroy()
		else:
			self.analysis_interval = 5000  # Default value
			self.text_analysis_interval.SetLabel('Default analysis interval = 5000 (milliseconds).')
		
		dialog.Destroy()

	def input_prediction_interval(self, event):
		dialog = wx.MessageDialog(
			self, 
			'Would you like to set a custom prediction interval?', 
			'Set Prediction Interval?', 
			wx.YES_NO | wx.ICON_QUESTION
		)
		
		if dialog.ShowModal() == wx.ID_YES:
			dialog.Destroy()
			dialog1 = wx.TextEntryDialog(self, 'Enter the custom prediction interval (will multiply by analysis interval):', 'Set Prediction Interval')
			
			if dialog1.ShowModal() == wx.ID_OK:
				try:
					self.prediction_interval = int(dialog1.GetValue())
					self.text_prediction_interval.SetLabel(f'Prediction interval set to {self.prediction_interval}.')
				except ValueError:
					wx.MessageBox('Invalid input! Please enter an integer.', 'Error', wx.OK | wx.ICON_ERROR)
					self.prediction_interval = 4  # Reset to default
					self.text_prediction_interval.SetLabel('Invalid input. Default prediction interval = 4 (4x analysis interval).')
			
			dialog1.Destroy()
		else:
			self.prediction_interval = 4  # Default value
			self.text_prediction_interval.SetLabel('Default prediction interval = 4 (4x analysis interval).')
		
		dialog.Destroy()

	def save_clip(self, frames, filename, fps):
		"""Saves a list of frames as a video file."""
		height, width, _ = frames[0].shape
		fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4
		out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

		for frame in frames:
			out.write(frame)

		out.release()

		# test_grooming_path = os.path.join(self.out_path, "mouse_grooming_clip.mp4")
		# print("Using test grooming path: ", test_grooming_path)
		# shutil.copy(test_grooming_path, filename)

		print(f"Saved clip: {filename}")

	def start_analysis(self, status):
		# Use GPU if available
		# device = "cuda" if torch.cuda.is_available() else "cpu"
		# print(f"Using {device} device")

		FPS = 30  # Set the FPS for capturing
		if self.analysis_interval is None:
			self.analysis_interval = 5000
		clip_length = int(self.analysis_interval//1000)  # Clip length in seconds
		frame_count = FPS * clip_length  # Total frames per clip
		output_dir = os.path.join(self.out_path, "captured_clips")  # Directory to save clips
		os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

		# Open webcam
		test_grooming_path = os.path.join(self.out_path, "mouse_grooming_clip_full.mp4")
		print("Using test grooming path: ", test_grooming_path)
		capture = cv2.VideoCapture(test_grooming_path)

		# Load the transformer model and the original encodings
		# !!! Want to load model in each worker or else they're all sharing one model
		# model, encodings = load_model(filepath=self.path_to_transformer)

		# model = model.to(device)
		# print("Encodings:", encodings)

		# capture = cv2.VideoCapture(0)
		if not capture.isOpened():
			print("Error: Could not open camera source.")
			return

		cv2.namedWindow("Live Stream", cv2.WINDOW_NORMAL)

		clip_number = 0

		# Thread safe ordered dictionary
		# Since dictionaries are persistent and python uses references,
		# you can share the memory between threads
		# Check MapReduce project for references
		manager = multiprocessing.Manager()
		status = manager.dict()

		# status["model"] = model
		# status["encodings"] = encodings
		status["analysis_queue"] = manager.Queue()

		# To make sure that the clips don't get added to master out of order
		current_clip_number = manager.Value("i", 0)
		shutdown = manager.Value("i", 0)

		# Just replace with threads to go back to multi-threaded
		# Issue with not enough sleep during behavior analysis, need to swap to processes
		# Uses cores in parallel instead of just taking advantage of I/O waiting
		num_analysis_processes = 4
		worker_process_list = []
		for i in range(num_analysis_processes):
			worker = Predict_Worker(status, self.get_predict_config(), current_clip_number, i, shutdown)
			worker_process = multiprocessing.Process(target=worker)
			worker_process_list.append(worker_process)
			worker_process.start()

		running = True

		print("current_clip_num:", current_clip_number.value)

		while running:
			time.sleep(0.1)
			print(f"\nCapturing {clip_length}-second clip {clip_number}...")
			frames = []

			start_time = time.time()
			while time.time() - start_time < clip_length:
				time.sleep(0.001)
				ret, frame = capture.read()
				if not ret:
					print("Error: Couldn't capture frame.")
					running = False
					break

				frames.append(frame)
				cv2.imshow("Live Stream", frame)  # Show live feed
				if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit if 'q' is pressed
					capture.release()
					cv2.destroyAllWindows()
					running = False
					break

			# Analyze behaviors takes a folder not a file, so every different clip needs its own folder ugggggg
			# Save the captured clip
			clip_folder = os.path.join(output_dir, f"clip_{clip_number}")
			os.makedirs(clip_folder, exist_ok=True)  # Create folder for each clip
			clip_filename = os.path.join(clip_folder, f"clip_{clip_number}.mp4")
			self.save_clip(frames, clip_filename, FPS)

			# Using threading instead of multiprocessing because apparently trying to perform GPU tasks across different CPUs introduces new issues
			task = {"clip_filename":clip_filename, "clip_folder": clip_folder, "clip_number": clip_number}
			status["analysis_queue"].put(task)
			clip_number += 1  # Increment clip count
			print("current_clip_num:", current_clip_number.value)
		
		while not clip_number == current_clip_number.value:
			# Recording is done
			# Wait for the rest of the clips to finish processing
			time.sleep(0.1)

		shutdown.value = 1

		# Release resources
		for worker_process in worker_process_list:
			worker_process.join()

		capture.release()
		cv2.destroyAllWindows()

	def get_predict_config(self):
		return {
			"path_to_transformer": self.path_to_transformer,
			"out_path": self.out_path,
			"analysis_interval": self.analysis_interval,
			"prediction_interval": self.prediction_interval,
			"behavior_mode": self.behavior_mode,
			"use_detector": self.use_detector,
			"detector_path": self.detector_path,
			"path_to_detector": self.path_to_detector,
			"detector_batch": self.detector_batch,
			"detection_threshold": self.detection_threshold,
			"animal_kinds": self.animal_kinds,
			"background_path": self.background_path,
			"model_path": self.model_path,
			"path_to_categorizer": self.path_to_categorizer,
			"framewidth": self.framewidth,
			"delta": self.delta,
			"decode_animalnumber": self.decode_animalnumber,
			"animal_number": self.animal_number,
			"autofind_t": self.autofind_t,
			"decode_t": self.decode_t,
			"t": self.t,
			"duration": self.duration,
			"decode_extraction": self.decode_extraction,
			"ex_start": self.ex_start,
			"ex_end": self.ex_end,
			"behaviornames_and_colors": self.behaviornames_and_colors,
			"dim_tconv": self.dim_tconv,
			"dim_conv": self.dim_conv,
			"channel": self.channel,
			"length": self.length,
			"animal_vs_bg": self.animal_vs_bg,
			"stable_illumination": self.stable_illumination,
			"animation_analyzer": self.animation_analyzer,
			"animal_to_include": self.animal_to_include,
			"behavior_to_include": self.behavior_to_include,
			"parameter_to_analyze": self.parameter_to_analyze,
			"include_bodyparts": self.include_bodyparts,
			"std": self.std,
			"uncertain": self.uncertain,
			"min_length": self.min_length,
			"show_legend": self.show_legend,
			"background_free": self.background_free,
			"black_background": self.black_background,
			"normalize_distance": self.normalize_distance,
			"social_distance": self.social_distance,
			"specific_behaviors": self.specific_behaviors,
			"correct_ID": self.correct_ID,
		}

class Live_Worker:
	def __init__(self, status, config, current_clip_number, worker_id, shutdown):
		# Dynamically assign all config keys as attributes
		for key, value in config.items():
			setattr(self, key, value)
		device = "cuda" if torch.cuda.is_available() else "cpu"
		# print(f"Using {device} device")
		self.status = status
		# The most recently processed clip across all workers
		self.current_clip_number = current_clip_number
		self.shutdown = shutdown
		self.worker_id = worker_id
		print("Worker " + str(worker_id) + " initialized!")

	def append_to_master_events_spreadsheet(self, clip_results_folder):
		"""Appends new behavior analysis results to a master Excel file, adjusting times to be continuous."""
		master_file = os.path.join(self.out_path, "all_events_master.xlsx")
		new_data_file = os.path.join(clip_results_folder, "all_events.xlsx")

		# Wait until new results file is available
		while not os.path.exists(new_data_file):
			print("Waiting for clip analysis results spreadsheet")
			time.sleep(0.01)

		new_data = pd.read_excel(new_data_file)

		# Set default offset
		time_offset = 0

		if os.path.exists(master_file):
			master_data = pd.read_excel(master_file)

			# Determine the time offset from the master spreadsheet
			# Note that pandas doesn't always read in the name, so the default for the first column would be Unnamed: 0
			if not master_data.empty:
				if "time/ID" in master_data.columns:
					time_offset = master_data["time/ID"].max()
				elif "Unnamed: 0" in master_data.columns:
					time_offset = master_data["Unnamed: 0"].max()
				print(f"Offsetting new data by {time_offset:.2f} seconds")

			# Rename index column if needed
			if "time/ID" not in new_data.columns and "Unnamed: 0" in new_data.columns:
				new_data.rename(columns={"Unnamed: 0": "time/ID"}, inplace=True)
				print("Renamed 'Unnamed: 0' to 'time/ID'")

			# Apply offset if column now exists
			if "time/ID" in new_data.columns:
				new_data["time/ID"] += time_offset
			else:
				print("Warning: 'time/ID' column not found in new data — skipping offset.")


			updated_data = pd.concat([master_data, new_data], ignore_index=True)
		else:
			print("Master spreadsheet not found. Creating a new one...")
			updated_data = new_data

		updated_data.to_excel(master_file, index=False)
		print(f"Updated master spreadsheet: {master_file}")

	
	def __call__(self):
			while self.shutdown.value == 0:
				time.sleep(1)
				# print("!!!!Queue Size: ", self.status["analysis_queue"].qsize())
				if not self.status["analysis_queue"].empty():
					print("!!!!Queue Size: ", self.status["analysis_queue"].qsize())
					print("Worker current clip number:", self.current_clip_number.value)
					task = self.status["analysis_queue"].get()
					clip_filename = task["clip_filename"]
					clip_folder = task["clip_folder"]
					print("Processing clip #", clip_filename)
					# Sleep to prevent busy waiting
					time.sleep(0.001)
					# Run behavior analysis on the captured clip
					self.path_to_videos = [clip_filename] # Set the path to the captured clip folder
					clip_results_folder = os.path.join(clip_folder, "results")
					os.makedirs(clip_results_folder, exist_ok=True)  # Create folder for results
					self.result_path = clip_results_folder  # Save output in the same directory

					# Issue is path_to_videos and result_path are self variables, fix with config
					self.analyze_behaviors(event=None)

					# Make the process wait so it doesn't add behaviors out of sequence
					while task["clip_number"] > self.current_clip_number.value:
						# print("Clip number: ", task["clip_number"])
						# print("Current clip number: ", self.current_clip_number.value)
						time.sleep(.1)

					# Append results to a master spreadsheet
					self.append_to_master_events_spreadsheet(clip_results_folder)

					self.current_clip_number.value = task["clip_number"] + 1

					print("Finished processing clip #", clip_filename)
				# print("Worker " + str(self.worker_id) + " waiting")
				# print("Shutdown value: " + str(self.shutdown.value))


	
	def analyze_behaviors(self, event):
		# Replaced all instances of self.path_to_videos and self.result_path with the path_to_videos and result_path arguments
		# Was messing up with threading because they were trying to overwrite shared instance paths

		if self.path_to_videos is None or self.result_path is None:

			wx.MessageBox('No input video(s) / result folder.','Error',wx.OK|wx.ICON_ERROR)

		else:

			if self.behavior_mode==3:

				if self.path_to_categorizer is None or self.path_to_detector is None:
					wx.MessageBox('You need to select a Categorizer / Detector.','Error',wx.OK|wx.ICON_ERROR)
				else:
					if len(self.animal_to_include)==0:
						self.animal_to_include=self.animal_kinds
					if self.detector_batch<=0:
						self.detector_batch=1
					if self.behavior_to_include[0]=='all':
						self.behavior_to_include=list(self.behaviornames_and_colors.keys())
					AAD=AnalyzeAnimalDetector()
					AAD.analyze_images_individuals(self.path_to_detector,self.path_to_videos,self.result_path,self.animal_kinds,path_to_categorizer=self.path_to_categorizer,
						generate=False,animal_to_include=self.animal_to_include,behavior_to_include=self.behavior_to_include,names_and_colors=self.behaviornames_and_colors,
						imagewidth=self.framewidth,dim_conv=self.dim_conv,channel=self.channel,detection_threshold=self.detection_threshold,uncertain=self.uncertain,
						background_free=self.background_free,black_background=self.black_background,social_distance=0)

			else:

				all_events={}
				event_data={}
				all_time=[]

				if self.use_detector:
					for animal_name in self.animal_kinds:
						all_events[animal_name]={}
					if len(self.animal_to_include)==0:
						self.animal_to_include=self.animal_kinds
					if self.detector_batch<=0:
						self.detector_batch=1

				if self.path_to_categorizer is None:
					self.behavior_to_include=[]
				else:
					if self.behavior_to_include[0]=='all':
						self.behavior_to_include=list(self.behaviornames_and_colors.keys())

				for i in self.path_to_videos:

					filename=os.path.splitext(os.path.basename(i))[0].split('_')
					if self.decode_animalnumber:
						if self.use_detector:
							self.animal_number={}
							number=[x[1:] for x in filename if len(x)>1 and x[0]=='n']
							for a,animal_name in enumerate(self.animal_kinds):
								self.animal_number[animal_name]=int(number[a])
						else:
							for x in filename:
								if len(x)>1:
									if x[0]=='n':
										self.animal_number=int(x[1:])
					if self.decode_t:
						for x in filename:
							if len(x)>1:
								if x[0]=='b':
									self.t=float(x[1:])
					if self.decode_extraction:
						for x in filename:
							if len(x)>2:
								if x[:2]=='xs':
									self.ex_start=int(x[2:])
								if x[:2]=='xe':
									self.ex_end=int(x[2:])

					if self.animal_number is None:
						if self.use_detector:
							self.animal_number={}
							for animal_name in self.animal_kinds:
								self.animal_number[animal_name]=1
						else:
							self.animal_number=1
				
					if self.path_to_categorizer is None:
						self.behavior_mode=0
						categorize_behavior=False
					else:
						categorize_behavior=True

					if self.use_detector is False:

						AA=AnalyzeAnimal()
						AA.prepare_analysis(i,self.result_path,self.animal_number,delta=self.delta,names_and_colors=self.behaviornames_and_colors,
							framewidth=self.framewidth,stable_illumination=self.stable_illumination,dim_tconv=self.dim_tconv,dim_conv=self.dim_conv,channel=self.channel,
							include_bodyparts=self.include_bodyparts,std=self.std,categorize_behavior=categorize_behavior,animation_analyzer=self.animation_analyzer,
							path_background=self.background_path,autofind_t=self.autofind_t,t=self.t,duration=self.duration,ex_start=self.ex_start,ex_end=self.ex_end,
							length=self.length,animal_vs_bg=self.animal_vs_bg)
						if self.behavior_mode==0:
							AA.acquire_information(background_free=self.background_free,black_background=self.black_background)
							AA.craft_data()
							interact_all=False
						else:
							AA.acquire_information_interact_basic(background_free=self.background_free,black_background=self.black_background)
							interact_all=True
						if self.path_to_categorizer is not None:
							AA.categorize_behaviors(self.path_to_categorizer,uncertain=self.uncertain,min_length=self.min_length)
						AA.annotate_video(self.behavior_to_include,show_legend=self.show_legend,interact_all=interact_all)
						AA.export_results(normalize_distance=self.normalize_distance,parameter_to_analyze=self.parameter_to_analyze)

						if self.path_to_categorizer is not None:
							for n in AA.event_probability:
								all_events[len(all_events)]=AA.event_probability[n]
							if len(all_time)<len(AA.all_time):
								all_time=AA.all_time

					else:

						AAD=AnalyzeAnimalDetector()
						AAD.prepare_analysis(self.path_to_detector,i,self.result_path,self.animal_number,self.animal_kinds,self.behavior_mode,
							names_and_colors=self.behaviornames_and_colors,framewidth=self.framewidth,dim_tconv=self.dim_tconv,dim_conv=self.dim_conv,channel=self.channel,
							include_bodyparts=self.include_bodyparts,std=self.std,categorize_behavior=categorize_behavior,animation_analyzer=self.animation_analyzer,
							t=self.t,duration=self.duration,length=self.length,social_distance=self.social_distance)
						if self.behavior_mode==1:
							AAD.acquire_information_interact_basic(batch_size=self.detector_batch,background_free=self.background_free,black_background=self.black_background)
						else:
							AAD.acquire_information(batch_size=self.detector_batch,background_free=self.background_free,black_background=self.black_background)
						if self.behavior_mode!=1:
							AAD.craft_data()
						if self.path_to_categorizer is not None:
							AAD.categorize_behaviors(self.path_to_categorizer,uncertain=self.uncertain,min_length=self.min_length)
						if self.correct_ID:
							AAD.correct_identity(self.specific_behaviors)
						AAD.annotate_video(self.animal_to_include,self.behavior_to_include,show_legend=self.show_legend)
						AAD.export_results(normalize_distance=self.normalize_distance,parameter_to_analyze=self.parameter_to_analyze)

						if self.path_to_categorizer is not None:
							for animal_name in self.animal_kinds:
								for n in AAD.event_probability[animal_name]:
									all_events[animal_name][len(all_events[animal_name])]=AAD.event_probability[animal_name][n]
							if len(all_time)<len(AAD.all_time):
								all_time=AAD.all_time
					
				if self.path_to_categorizer is not None:

					max_length=len(all_time)

					if self.use_detector is False:

						for n in all_events:
							event_data[len(event_data)]=all_events[n]+[['NA',-1]]*(max_length-len(all_events[n]))
						all_events_df=pd.DataFrame(event_data,index=all_time)
						all_events_df.to_excel(os.path.join(self.result_path,'all_events.xlsx'),float_format='%.2f',index_label='time/ID')
						plot_events(self.result_path,event_data,all_time,self.behaviornames_and_colors,self.behavior_to_include,width=0,height=0)
						folders=[i for i in os.listdir(self.result_path) if os.path.isdir(os.path.join(self.result_path,i))]
						folders.sort()
						for behavior_name in self.behaviornames_and_colors:
							all_summary=[]
							for folder in folders:
								individual_summary=os.path.join(self.result_path,folder,behavior_name,'all_summary.xlsx')
								if os.path.exists(individual_summary):
									all_summary.append(pd.read_excel(individual_summary))
							if len(all_summary)>=1:
								all_summary=pd.concat(all_summary,ignore_index=True)
								all_summary.to_excel(os.path.join(self.result_path,behavior_name+'_summary.xlsx'),float_format='%.2f',index_label='ID/parameter')

					else:

						for animal_name in self.animal_to_include:
							for n in all_events[animal_name]:
								event_data[len(event_data)]=all_events[animal_name][n]+[['NA',-1]]*(max_length-len(all_events[animal_name][n]))
							event_data[len(event_data)]=[['NA',-1]]*max_length
						del event_data[len(event_data)-1]
						all_events_df=pd.DataFrame(event_data,index=all_time)
						all_events_df.to_excel(os.path.join(self.result_path,'all_events.xlsx'),float_format='%.2f',index_label='time/ID')
						plot_events(self.result_path,event_data,all_time,self.behaviornames_and_colors,self.behavior_to_include,width=0,height=0)
						folders=[i for i in os.listdir(self.result_path) if os.path.isdir(os.path.join(self.result_path,i))]
						folders.sort()
						for animal_name in self.animal_kinds:
							for behavior_name in self.behaviornames_and_colors:
								all_summary=[]
								for folder in folders:
									individual_summary=os.path.join(self.result_path,folder,behavior_name,animal_name+'_all_summary.xlsx')
									if os.path.exists(individual_summary):
										all_summary.append(pd.read_excel(individual_summary))
								if len(all_summary)>=1:
									all_summary=pd.concat(all_summary,ignore_index=True)
									all_summary.to_excel(os.path.join(self.result_path,animal_name+'_'+behavior_name+'_summary.xlsx'),float_format='%.2f',index_label='ID/parameter')

			print('Analysis completed!')

class WindowLv2_LiveAnalysisSettings(wx.Frame):

	'''
	The 'Live Analysis Settings' window
	'''

	def __init__(self,title):

		super(WindowLv2_LiveAnalysisSettings,self).__init__(parent=None,title=title,size=(1000,560))
		self.path_to_clip=None # path to clip if testing
		self.out_path=None # the folder to store output
		self.analysis_interval=None # the interval between live behavior analysis in milliseconds
		self.video_input_mode = 'camera' # variable for toggling between camera and clip video source
		self.live_camera = -1 # camera number
		self.num_workers = 5 # number of worker processes

		#COPY
		self.behavior_mode=0 # 0--non-interactive, 1--interactive basic, 2--interactive advanced, 3--static images
		self.use_detector=False # whether the Detector is used
		self.detector_path=None # the 'LabGym/detectors' folder, which stores all the trained Detectors
		self.path_to_detector=None # path to the Detector
		self.detector_batch=1 # for batch processing use if GPU is available
		self.detection_threshold=0 # only for 'static images' behavior mode
		self.animal_kinds=[] # the total categories of animals / objects in a Detector
		self.background_path=None # if not None, load background images from path in 'background subtraction' detection method
		self.model_path=None # the 'LabGym/models' folder, which stores all the trained Categorizers
		self.path_to_categorizer=None # path to the Categorizer
		self.path_to_videos=None # path to a batch of videos for analysis
		self.result_path=None # the folder for storing analysis outputs
		self.framewidth=None # if not None, will resize the video frame keeping the original w:h ratio
		self.delta=10000 # the fold changes in illumination that determines the optogenetic stimulation onset
		self.decode_animalnumber=False # whether to decode animal numbers from '_nn_' in video file names
		self.animal_number=1 # the number of animals / objects in a video
		self.autofind_t=False # whether to find stimulation onset automatically (only for optogenetics)
		self.decode_t=False # whether to decode start_t from '_bt_' in video file names
		self.t=0 # the start_t for analysis
		self.duration=0 # the duration of the analysis
		self.decode_extraction=False # whether to decode time windows for background extraction from '_xst_' and '_xet_' in video file names
		self.ex_start=0 # start time for background extraction
		self.ex_end=None # end time for background extraction
		self.behaviornames_and_colors={} # behavior names in the Categorizer and their representative colors for annotation
		self.dim_tconv=8 # input dimension for Animation Analyzer in Categorizer
		self.dim_conv=8 # input dimension for Pattern Recognizer in Categorizer
		self.channel=1 # input channel for Animation Analyzer, 1--gray scale, 3--RGB scale
		self.length=15 # input time step for Animation Analyzer, also the duration / length for a behavior example
		self.animal_vs_bg=0 # 0: animals birghter than the background; 1: animals darker than the background; 2: hard to tell
		self.stable_illumination=True # whether the illumination in videos is stable
		self.animation_analyzer=True # whether to include Animation Analyzer in the Categorizers
		self.animal_to_include=[] # the animals / obejcts that will be annotated in the annotated videos / behavior plots
		self.behavior_to_include=['all'] # behaviors that will be annotated in the annotated videos / behavior plots
		self.parameter_to_analyze=[] # quantitative measures that will be included in the quantification
		self.include_bodyparts=False # whether to include body parts in the pattern images
		self.std=0 # a value between 0 and 255, higher value, less body parts will be included in the pattern images
		self.uncertain=0 # a threshold between the highest the 2nd highest probablity of behaviors to determine if output an 'NA' in behavior classification
		self.min_length=None # the minimum length (in frames) a behavior should last, can be used to filter out the brief false positives
		self.show_legend=True # whether to show legend of behavior names in the annotated videos
		self.background_free=True # whether to include background in animations
		self.black_background=True # whether to set background black
		self.normalize_distance=True # whether to normalize the distance (in pixel) to the animal contour area
		self.social_distance=0 # a threshold (folds of size of a single animal) on whether to include individuals that are not main character in behavior examples
		self.specific_behaviors={} # sex or identity-specific behaviors
		self.correct_ID=False # whether to use sex or identity-specific behaviors to guide ID correction when ID switching is likely to happen
		#END COPY
	
		self.display_window()


	def display_window(self):

		panel=wx.Panel(self)
		boxsizer=wx.BoxSizer(wx.VERTICAL)

		#COPY
		module_selectcategorizer=wx.BoxSizer(wx.HORIZONTAL)
		button_selectcategorizer=wx.Button(panel,label='Select a Categorizer for\nbehavior classification',size=(300,40))
		button_selectcategorizer.Bind(wx.EVT_BUTTON,self.select_categorizer)
		wx.Button.SetToolTip(button_selectcategorizer,'The fps of the videos to analyze should match that of the selected Categorizer. Uncertain level determines the threshold for the Categorizer to output an ‘NA’ for behavioral classification. See Extended Guide for details.')
		self.text_selectcategorizer=wx.StaticText(panel,label='Default: no behavior classification, just track animals and quantify motion kinematcis.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_selectcategorizer.Add(button_selectcategorizer,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_selectcategorizer.Add(self.text_selectcategorizer,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,10,0)
		boxsizer.Add(module_selectcategorizer,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		# REPLACE WITH CLIP

		# module_inputvideos=wx.BoxSizer(wx.HORIZONTAL)
		# button_inputvideos=wx.Button(panel,label='Select the video(s) / image(s)\nfor behavior analysis',size=(300,40))
		# button_inputvideos.Bind(wx.EVT_BUTTON,self.select_videos)
		# wx.Button.SetToolTip(button_inputvideos,'Select one or more videos / images for a behavior analysis batch. If analyzing videos, one analysis batch will yield one raster plot showing the behavior events of all the animals in all selected videos. For "Static images" mode, each annotated images will be in this folder. See Extended Guide for details.')
		# self.text_inputvideos=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		# module_inputvideos.Add(button_inputvideos,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		# module_inputvideos.Add(self.text_inputvideos,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		# boxsizer.Add(module_inputvideos,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		# boxsizer.Add(0,5,0)

		# module_outputfolder=wx.BoxSizer(wx.HORIZONTAL)
		# button_outputfolder=wx.Button(panel,label='Select a folder to store\nthe analysis results',size=(300,40))
		# button_outputfolder.Bind(wx.EVT_BUTTON,self.select_outpath)
		# wx.Button.SetToolTip(button_outputfolder,'If analyzing videos, will create a subfolder for each video in the selected folder. Each subfolder is named after the file name of the video and stores the detailed analysis results for this video. For "Static images" mode, all results will be in this folder. See Extended Guide for details.')
		# self.text_outputfolder=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		# module_outputfolder.Add(button_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		# module_outputfolder.Add(self.text_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		# boxsizer.Add(module_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		# boxsizer.Add(0,5,0)

		module_detection=wx.BoxSizer(wx.HORIZONTAL)
		button_detection=wx.Button(panel,label='Specify the method to\ndetect animals or objects',size=(300,40))
		button_detection.Bind(wx.EVT_BUTTON,self.select_method)
		wx.Button.SetToolTip(button_detection,'Background subtraction-based method is accurate and fast but requires static background and stable illumination in videos; Detectors-based method is accurate and versatile in any recording settings but is slow. See Extended Guide for details.')
		self.text_detection=wx.StaticText(panel,label='Default: Background subtraction-based method.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_detection.Add(button_detection,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_detection.Add(self.text_detection,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_detection,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		# LEAVE DEFAULT AT 0

		# module_startanalyze=wx.BoxSizer(wx.HORIZONTAL)
		# button_startanalyze=wx.Button(panel,label='Specify when the analysis\nshould begin (unit: second)',size=(300,40))
		# button_startanalyze.Bind(wx.EVT_BUTTON,self.specify_timing)
		# wx.Button.SetToolTip(button_startanalyze,'Enter a beginning time point for all videos in one analysis batch or use "Decode from filenames" to let LabGym decode the different beginning time for different videos. See Extended Guide for details.')
		# self.text_startanalyze=wx.StaticText(panel,label='Default: at the beginning of the video(s).',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		# module_startanalyze.Add(button_startanalyze,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		# module_startanalyze.Add(self.text_startanalyze,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		# boxsizer.Add(module_startanalyze,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		# boxsizer.Add(0,5,0)

		# LEAVE DEFAULT TILL END

		# module_duration=wx.BoxSizer(wx.HORIZONTAL)
		# button_duration=wx.Button(panel,label='Specify the analysis duration\n(unit: second)',size=(300,40))
		# button_duration.Bind(wx.EVT_BUTTON,self.input_duration)
		# wx.Button.SetToolTip(button_duration,'The duration is the same for all the videos in a same analysis batch.')
		# self.text_duration=wx.StaticText(panel,label='Default: from the specified beginning time to the end of a video.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		# module_duration.Add(button_duration,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		# module_duration.Add(self.text_duration,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		# boxsizer.Add(module_duration,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		# boxsizer.Add(0,5,0)

		# ONLY WORKS FOR ONE ANIMAL FOR NOW SO LEAVE DEFAULT AT 1

		# module_animalnumber=wx.BoxSizer(wx.HORIZONTAL)
		# button_animalnumber=wx.Button(panel,label='Specify the number of animals\nin a video',size=(300,40))
		# button_animalnumber.Bind(wx.EVT_BUTTON,self.specify_animalnumber)
		# wx.Button.SetToolTip(button_animalnumber,'Enter a number for all videos in one analysis batch or use "Decode from filenames" to let LabGym decode the different animal number for different videos. See Extended Guide for details.')
		# self.text_animalnumber=wx.StaticText(panel,label='Default: 1.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		# module_animalnumber.Add(button_animalnumber,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		# module_animalnumber.Add(self.text_animalnumber,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		# boxsizer.Add(module_animalnumber,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		# boxsizer.Add(0,5,0)

		module_selectbehaviors=wx.BoxSizer(wx.HORIZONTAL)
		button_selectbehaviors=wx.Button(panel,label='Select the behaviors for\nannotations and plots',size=(300,40))
		button_selectbehaviors.Bind(wx.EVT_BUTTON,self.select_behaviors)
		wx.Button.SetToolTip(button_selectbehaviors,'The behavior categories are determined by the selected Categorizer. Select which behaviors to show in the annotated videos / images and the raster plot (only for videos). See Extended Guide for details.')
		self.text_selectbehaviors=wx.StaticText(panel,label='Default: No Categorizer selected, no behavior selected.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_selectbehaviors.Add(button_selectbehaviors,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_selectbehaviors.Add(self.text_selectbehaviors,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_selectbehaviors,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_selectparameters=wx.BoxSizer(wx.HORIZONTAL)
		button_selectparameters=wx.Button(panel,label='Select the quantitative measurements\nfor each behavior',size=(300,40))
		button_selectparameters.Bind(wx.EVT_BUTTON,self.select_parameters)
		wx.Button.SetToolTip(button_selectparameters,'If select "not to normalize distances", all distances will be output in pixels. If select "normalize distances", all distances will be normalized to the animal size. See Extended Guide for details.')
		self.text_selectparameters=wx.StaticText(panel,label='Default: none.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_selectparameters.Add(button_selectparameters,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_selectparameters.Add(self.text_selectparameters,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_selectparameters,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		#END COPY

		module_videosource_toggle=wx.BoxSizer(wx.HORIZONTAL)
		self.toggle_input_button = wx.Button(panel, label='Switch to File Input for Testing', size=(200, 30))
		self.toggle_input_button.Bind(wx.EVT_BUTTON, self.select_video_input_mode)
		module_videosource_toggle.Add(self.toggle_input_button,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		self.text_toggle_input_button=wx.StaticText(panel,label='Option to select a video clip for mock live analysis.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_videosource_toggle.Add(self.text_toggle_input_button, 0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_videosource_toggle,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_videosource=wx.BoxSizer(wx.HORIZONTAL)
		self.button_videosource=wx.Button(panel,label='Select camera source \nfor live analysis',size=(300,40))
		self.button_videosource.Bind(wx.EVT_BUTTON,self.handle_video_input)
		wx.Button.SetToolTip(self.button_videosource,'Select the camera for live analysis.')
		self.text_videosource=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_videosource.Add(self.button_videosource,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_videosource.Add(self.text_videosource,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_videosource,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_num_workers=wx.BoxSizer(wx.HORIZONTAL)
		self.button_num_workers=wx.Button(panel,label='Select the number of workers \nfor live analysis',size=(300,40))
		self.button_num_workers.Bind(wx.EVT_BUTTON,self.select_num_workers)
		wx.Button.SetToolTip(self.button_num_workers,'Select the number of worker processes for live analysis (each worker can process one clip on one core at a time).')
		self.text_num_workers=wx.StaticText(panel,label="Number of workers: " + str(self.num_workers),style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_num_workers.Add(self.button_num_workers,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_num_workers.Add(self.text_num_workers,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_num_workers,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_analysis_interval=wx.BoxSizer(wx.HORIZONTAL)
		button_analysis_interval=wx.Button(panel,label='Specify the interval of capture (in milliseconds)\nfor live behavior analysis',size=(300,40))
		button_analysis_interval.Bind(wx.EVT_BUTTON,self.input_analysis_interval)
		wx.Button.SetToolTip(button_analysis_interval,'Set a custom number of milliseconds for the clip length for each behavior analysis.')
		self.text_analysis_interval=wx.StaticText(panel,label='Default: 5000 (5 seconds).',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_analysis_interval.Add(button_analysis_interval,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_analysis_interval.Add(self.text_analysis_interval,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_analysis_interval,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_outputfolder=wx.BoxSizer(wx.HORIZONTAL)
		button_outputfolder=wx.Button(panel,label='Select the folder to store\nthe analyzed behaviors',size=(300,40))
		button_outputfolder.Bind(wx.EVT_BUTTON,self.select_outputpath)
		wx.Button.SetToolTip(button_outputfolder,'In this folder there will be a file containing the live analysis output.')
		self.text_outputfolder=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_outputfolder.Add(button_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_outputfolder.Add(self.text_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		button_start_analysis=wx.Button(panel,label='Start analysis',size=(300,40))
		button_start_analysis.Bind(wx.EVT_BUTTON,self.start_analysis)
		wx.Button.SetToolTip(button_start_analysis,'Start live analysis.')
		boxsizer.Add(0,5,0)
		boxsizer.Add(button_start_analysis,0,wx.RIGHT|wx.ALIGN_RIGHT,90)
		boxsizer.Add(0,10,0)

		panel.SetSizer(boxsizer)

		self.Centre()
		self.Show(True)

	#COPY

	def select_video_input_mode(self, event):
		if self.video_input_mode == 'camera':
			self.video_input_mode = 'file'
			self.button_videosource.SetLabel('Select a video clip \nfor mock live analysis')
			self.button_videosource.SetToolTip('Select a video file for mock live analysis.')
			self.toggle_input_button.SetLabel('Switch to Camera Input')
		else:
			self.video_input_mode = 'camera'
			self.button_videosource.SetLabel('Select camera source \nfor live analysis')
			self.button_videosource.SetToolTip('Select the camera for live analysis..')
			self.toggle_input_button.SetLabel('Switch to File Input')

	def handle_video_input(self, event):
		if self.video_input_mode == 'camera':
			self.select_videosource(event)
		else:
			self.select_input_file(event)

	def select_input_file(self, event):
		with wx.FileDialog(self, "Select video file", wildcard="Video files (*.mp4;*.avi)|*.mp4;*.avi",
						style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
			if fileDialog.ShowModal() == wx.ID_OK:
				path = fileDialog.GetPath()
				self.path_to_clip = path
				self.text_videosource.SetLabel(f"Selected file: {path}")


	def select_method(self,event):

		if self.behavior_mode<=1:
			methods=['Subtract background (fast but requires static background & stable illumination)','Use trained Detectors (versatile but slow)']
		else:
			methods=['Use trained Detectors (versatile but slow)']

		dialog=wx.SingleChoiceDialog(self,message='How to detect the animals?',caption='Detection methods',choices=methods)

		if dialog.ShowModal()==wx.ID_OK:
			method=dialog.GetStringSelection()

			if method=='Subtract background (fast but requires static background & stable illumination)':

				self.use_detector=False

				contrasts=['Animal brighter than background','Animal darker than background','Hard to tell']
				dialog1=wx.SingleChoiceDialog(self,message='Select the scenario that fits your videos best',caption='Which fits best?',choices=contrasts)

				if dialog1.ShowModal()==wx.ID_OK:
					contrast=dialog1.GetStringSelection()
					if contrast=='Animal brighter than background':
						self.animal_vs_bg=0
					elif contrast=='Animal darker than background':
						self.animal_vs_bg=1
					else:
						self.animal_vs_bg=2
					dialog2=wx.MessageDialog(self,'Load an existing background from a folder?\nSelect "No" if dont know what it is.','(Optional) load existing background?',wx.YES_NO|wx.ICON_QUESTION)
					if dialog2.ShowModal()==wx.ID_YES:
						dialog3=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE)
						if dialog3.ShowModal()==wx.ID_OK:
							self.background_path=dialog3.GetPath()
						dialog3.Destroy()
					else:
						self.background_path=None
						if self.animal_vs_bg!=2:
							dialog3=wx.MessageDialog(self,'Unstable illumination in the video?\nSelect "Yes" if dont know what it is.','(Optional) unstable illumination?',wx.YES_NO|wx.ICON_QUESTION)
							if dialog3.ShowModal()==wx.ID_YES:
								self.stable_illumination=False
							else:
								self.stable_illumination=True
							dialog3.Destroy()
					dialog2.Destroy()

					if self.background_path is None:
						ex_methods=['Use the entire duration (default but NOT recommended)','Decode from filenames: "_xst_" and "_xet_"','Enter two time points']
						dialog2=wx.SingleChoiceDialog(self,message='Specify the time window for background extraction',caption='Time window for background extraction',choices=ex_methods)
						if dialog2.ShowModal()==wx.ID_OK:
							ex_method=dialog2.GetStringSelection()
							if ex_method=='Use the entire duration (default but NOT recommended)':
								self.decode_extraction=False
								if self.animal_vs_bg==0:
									self.text_detection.SetLabel('Background subtraction: animal brighter, using the entire duration.')
								elif self.animal_vs_bg==1:
									self.text_detection.SetLabel('Background subtraction: animal darker, using the entire duration.')
								else:
									self.text_detection.SetLabel('Background subtraction: animal partially brighter/darker, using the entire duration.')
							elif ex_method=='Decode from filenames: "_xst_" and "_xet_"':
								self.decode_extraction=True
								if self.animal_vs_bg==0:
									self.text_detection.SetLabel('Background subtraction: animal brighter, using time window decoded from filenames "_xst_" and "_xet_".')
								elif self.animal_vs_bg==1:
									self.text_detection.SetLabel('Background subtraction: animal darker, using time window decoded from filenames "_xst_" and "_xet_".')
								else:
									self.text_detection.SetLabel('Background subtraction: animal partially brighter/darker, using time window decoded from filenames "_xst_" and "_xet_".')
							else:
								self.decode_extraction=False
								dialog3=wx.NumberEntryDialog(self,'Enter the start time','The unit is second:','Start time for background extraction',0,0,100000000000000)
								if dialog3.ShowModal()==wx.ID_OK:
									self.ex_start=int(dialog3.GetValue())
								dialog3.Destroy()
								dialog3=wx.NumberEntryDialog(self,'Enter the end time','The unit is second:','End time for background extraction',0,0,100000000000000)
								if dialog3.ShowModal()==wx.ID_OK:
									self.ex_end=int(dialog3.GetValue())
									if self.ex_end==0:
										self.ex_end=None
								dialog3.Destroy()
								if self.animal_vs_bg==0:
									if self.ex_end is None:
										self.text_detection.SetLabel('Background subtraction: animal brighter, using time window (in seconds) from '+str(self.ex_start)+' to the end.')
									else:
										self.text_detection.SetLabel('Background subtraction: animal brighter, using time window (in seconds) from '+str(self.ex_start)+' to '+str(self.ex_end)+'.')
								elif self.animal_vs_bg==1:
									if self.ex_end is None:
										self.text_detection.SetLabel('Background subtraction: animal darker, using time window (in seconds) from '+str(self.ex_start)+' to the end.')
									else:
										self.text_detection.SetLabel('Background subtraction: animal darker, using time window (in seconds) from '+str(self.ex_start)+' to '+str(self.ex_end)+'.')
								else:
									if self.ex_end is None:
										self.text_detection.SetLabel('Background subtraction: animal partially brighter/darker, using time window (in seconds) from '+str(self.ex_start)+' to the end.')
									else:
										self.text_detection.SetLabel('Background subtraction: animal partially brighter/darker, using time window (in seconds) from '+str(self.ex_start)+' to '+str(self.ex_end)+'.')
						dialog2.Destroy()

				dialog1.Destroy()

			else:

				self.use_detector=True
				self.animal_number={}
				self.detector_path=os.path.join(the_absolute_current_path,'detectors')

				detectors=[i for i in os.listdir(self.detector_path) if os.path.isdir(os.path.join(self.detector_path,i))]
				if '__pycache__' in detectors:
					detectors.remove('__pycache__')
				if '__init__' in detectors:
					detectors.remove('__init__')
				if '__init__.py' in detectors:
					detectors.remove('__init__.py')
				detectors.sort()
				if 'Choose a new directory of the Detector' not in detectors:
					detectors.append('Choose a new directory of the Detector')

				dialog1=wx.SingleChoiceDialog(self,message='Select a Detector for animal detection',caption='Select a Detector',choices=detectors)
				if dialog1.ShowModal()==wx.ID_OK:
					detector=dialog1.GetStringSelection()
					if detector=='Choose a new directory of the Detector':
						dialog2=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE)
						if dialog2.ShowModal()==wx.ID_OK:
							self.path_to_detector=dialog2.GetPath()
						dialog2.Destroy()
					else:
						self.path_to_detector=os.path.join(self.detector_path,detector)
					with open(os.path.join(self.path_to_detector,'model_parameters.txt')) as f:
						model_parameters=f.read()
					animal_names=json.loads(model_parameters)['animal_names']
					if len(animal_names)>1:
						dialog2=wx.MultiChoiceDialog(self,message='Specify which animals/objects involved in analysis',caption='Animal/Object kind',choices=animal_names)
						if dialog2.ShowModal()==wx.ID_OK:
							self.animal_kinds=[animal_names[i] for i in dialog2.GetSelections()]
						else:
							self.animal_kinds=animal_names
						dialog2.Destroy()
					else:
						self.animal_kinds=animal_names
					self.animal_to_include=self.animal_kinds
					if self.behavior_mode>=3:
						dialog2=wx.NumberEntryDialog(self,"Enter the Detector's detection threshold (0~100%)","The higher detection threshold,\nthe higher detection accuracy,\nbut the lower detection sensitivity.\nEnter 0 if don't know how to set.",'Detection threshold',0,0,100)
						if dialog2.ShowModal()==wx.ID_OK:
							detection_threshold=dialog2.GetValue()
							self.detection_threshold=detection_threshold/100
						self.text_detection.SetLabel('Detector: '+detector+' (detection threshold: '+str(detection_threshold)+'%); The animals/objects: '+str(self.animal_kinds)+'.')
						dialog2.Destroy()
					else:
						for animal_name in self.animal_kinds:
							self.animal_number[animal_name]=1
						# Not needed because you don't select the number of animals, throws error in console
						# self.text_animalnumber.SetLabel('The number of '+str(self.animal_kinds)+' is: '+str(list(self.animal_number.values()))+'.')
						self.text_detection.SetLabel('Detector: '+detector+'; '+'The animals/objects: '+str(self.animal_kinds)+'.')
				dialog1.Destroy()

				if self.behavior_mode<3:
					if torch.cuda.is_available():
						dialog1=wx.NumberEntryDialog(self,'Enter the batch size for faster processing','GPU is available in this device for Detectors.\nYou may use batch processing for faster speed.','Batch size',1,1,100)
						if dialog1.ShowModal()==wx.ID_OK:
							self.detector_batch=int(dialog1.GetValue())
						else:
							self.detector_batch=1
						dialog1.Destroy()

		dialog.Destroy()

	def select_categorizer(self,event):

		if self.model_path is None:
			self.model_path=os.path.join(the_absolute_current_path,'models')

		categorizers=[i for i in os.listdir(self.model_path) if os.path.isdir(os.path.join(self.model_path,i))]
		if '__pycache__' in categorizers:
			categorizers.remove('__pycache__')
		if '__init__' in categorizers:
			categorizers.remove('__init__')
		if '__init__.py' in categorizers:
			categorizers.remove('__init__.py')
		categorizers.sort()
		if 'No behavior classification, just track animals and quantify motion kinematics' not in categorizers:
			categorizers.append('No behavior classification, just track animals and quantify motion kinematics')
		if 'Choose a new directory of the Categorizer' not in categorizers:
			categorizers.append('Choose a new directory of the Categorizer')

		dialog=wx.SingleChoiceDialog(self,message='Select a Categorizer for behavior classification',caption='Select a Categorizer',choices=categorizers)

		if dialog.ShowModal()==wx.ID_OK:
			categorizer=dialog.GetStringSelection()
			if categorizer=='Choose a new directory of the Categorizer':
				dialog1=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE)
				if dialog1.ShowModal()==wx.ID_OK:
					self.path_to_categorizer=dialog1.GetPath()
				dialog1.Destroy()
				dialog1=wx.NumberEntryDialog(self,"Enter the Categorizer's uncertainty level (0~100%)","If probability difference between\n1st- and 2nd-likely behaviors\nis less than uncertainty,\nclassfication outputs an 'NA'. Enter 0 if don't know how to set.",'Uncertainty level',0,0,100)
				if dialog1.ShowModal()==wx.ID_OK:
					uncertain=dialog1.GetValue()
					self.uncertain=uncertain/100
				else:
					uncertain=0
					self.uncertain=0
				dialog1.Destroy()
				if self.behavior_mode<3:
					dialog1=wx.MessageDialog(self,"Set a minimum length (in frames) for a behavior episode\nto output 'NA' if the duration of a identified behavior\nis shorter than the minimun length?",'Minimum length?',wx.YES_NO|wx.ICON_QUESTION)
					if dialog1.ShowModal()==wx.ID_YES:
						dialog2=wx.NumberEntryDialog(self,'Enter the minimun length (in frames)',"If the duration of a identified behavior\nis shorter than the minimun length,\nthe behavior categorization will output as 'NA'.",'Minimum length',2,1,10000)
						if dialog2.ShowModal()==wx.ID_OK:
							self.min_length=int(dialog2.GetValue())
							if self.min_length<2:
								self.min_length=2
						else:
							self.min_length=None
						dialog2.Destroy()
					else:
						self.min_length=None
					dialog1.Destroy()
				if self.min_length is None:
					self.text_selectcategorizer.SetLabel('The path to the Categorizer is: '+self.path_to_categorizer+' with uncertainty of '+str(uncertain)+'%.')
				else:
					self.text_selectcategorizer.SetLabel('The path to the Categorizer is: '+self.path_to_categorizer+' with uncertainty of '+str(uncertain)+'%; minimun length of '+str(self.min_length)+'.')
				self.text_selectbehaviors.SetLabel('All the behaviors in the selected Categorizer with default colors.')
			elif categorizer=='No behavior classification, just track animals and quantify motion kinematics':
				self.path_to_categorizer=None
				self.behavior_mode=0
				dialog1=wx.NumberEntryDialog(self,'Specify a time window used for measuring\nmotion kinematics of the tracked animals','Enter the number of\nframes (minimum=3):','Time window for calculating kinematics',15,1,100000000000000)
				if dialog1.ShowModal()==wx.ID_OK:
					self.length=int(dialog1.GetValue())
					if self.length<3:
						self.length=3
				dialog1.Destroy()
				self.text_selectcategorizer.SetLabel('No behavior classification; the time window to measure kinematics of tracked animals is: '+str(self.length)+' frames.')
				self.text_selectbehaviors.SetLabel('No behavior classification. Just track animals and quantify motion kinematics.')
			else:
				self.path_to_categorizer=os.path.join(self.model_path,categorizer)
				dialog1=wx.NumberEntryDialog(self,"Enter the Categorizer's uncertainty level (0~100%)","If probability difference between\n1st- and 2nd-likely behaviors\nis less than uncertainty,\nclassfication outputs an 'NA'.",'Uncertainty level',0,0,100)
				if dialog1.ShowModal()==wx.ID_OK:
					uncertain=dialog1.GetValue()
					self.uncertain=uncertain/100
				else:
					uncertain=0
					self.uncertain=0
				dialog1.Destroy()
				if self.behavior_mode<3:
					dialog1=wx.MessageDialog(self,"Set a minimum length (in frames) for a behavior episode\nto output 'NA' if the duration of a identified behavior\nis shorter than the minimun length?",'Minimum length?',wx.YES_NO|wx.ICON_QUESTION)
					if dialog1.ShowModal()==wx.ID_YES:
						dialog2=wx.NumberEntryDialog(self,'Enter the minimun length (in frames)',"If the duration of a identified behavior\nis shorter than the minimun length,\nthe behavior categorization will output as 'NA'.",'Minimum length',2,1,10000)
						if dialog2.ShowModal()==wx.ID_OK:
							self.min_length=int(dialog2.GetValue())
							if self.min_length<2:
								self.min_length=2
						else:
							self.min_length=None
						dialog2.Destroy()
					else:
						self.min_length=None
					dialog1.Destroy()
				if self.min_length is None:
					self.text_selectcategorizer.SetLabel('Categorizer: '+categorizer+' with uncertainty of '+str(uncertain)+'%.')
				else:
					self.text_selectcategorizer.SetLabel('Categorizer: '+categorizer+' with uncertainty of '+str(uncertain)+'%; minimun length of '+str(self.min_length)+'.')
				self.text_selectbehaviors.SetLabel('All the behaviors in the selected Categorizer with default colors.')

			if self.path_to_categorizer is not None:

				parameters=pd.read_csv(os.path.join(self.path_to_categorizer,'model_parameters.txt'))
				complete_colors=list(mpl.colors.cnames.values())
				colors=[]
				for c in complete_colors:
					colors.append(['#ffffff',c])
				self.behaviornames_and_colors={}

				for behavior_name in list(parameters['classnames']):
					index=list(parameters['classnames']).index(behavior_name)
					if index<len(colors):
						self.behaviornames_and_colors[behavior_name]=colors[index]
					else:
						self.behaviornames_and_colors[behavior_name]=['#ffffff','#ffffff']

				if 'dim_conv' in parameters:
					self.dim_conv=int(parameters['dim_conv'][0])
				if 'dim_tconv' in parameters:
					self.dim_tconv=int(parameters['dim_tconv'][0])
				self.channel=int(parameters['channel'][0])
				self.length=int(parameters['time_step'][0])
				if self.length<3:
					self.length=3
				categorizer_type=int(parameters['network'][0])
				if categorizer_type==2:
					self.animation_analyzer=True
				else:
					self.animation_analyzer=False
				if int(parameters['inner_code'][0])==0:
					self.include_bodyparts=True
				else:
					self.include_bodyparts=False
				self.std=int(parameters['std'][0])
				if int(parameters['background_free'][0])==0:
					self.background_free=True
				else:
					self.background_free=False
				if 'behavior_kind' in parameters:
					self.behavior_mode=int(parameters['behavior_kind'][0])
				else:
					self.behavior_mode=0
				if self.behavior_mode==2:
					self.social_distance=int(parameters['social_distance'][0])
					if self.social_distance==0:
						self.social_distance=float('inf')
					self.text_detection.SetLabel('Only Detector-based detection method is available for the selected Categorizer.')
				if self.behavior_mode==3:
					self.text_detection.SetLabel('Only Detector-based detection method is available for the selected Categorizer.')
					self.text_startanalyze.SetLabel('No need to specify this since the selected behavior mode is "Static images".')
					self.text_duration.SetLabel('No need to specify this since the selected behavior mode is "Static images".')
					self.text_animalnumber.SetLabel('No need to specify this since the selected behavior mode is "Static images".')
					self.text_selectparameters.SetLabel('No need to specify this since the selected behavior mode is "Static images".')
				if 'black_background' in parameters:
					if int(parameters['black_background'][0])==1:
						self.black_background=False

		dialog.Destroy()

	def select_behaviors(self,event):

		if self.path_to_categorizer is None:

			wx.MessageBox('No Categorizer selected! The behavior names are listed in the Categorizer.','Error',wx.OK|wx.ICON_ERROR)

		else:

			if len(self.animal_kinds)>1:
				dialog=wx.MultiChoiceDialog(self,message='Specify which animals/objects to annotate',caption='Animal/Object to annotate',choices=self.animal_kinds)
				if dialog.ShowModal()==wx.ID_OK:
					self.animal_to_include=[self.animal_kinds[i] for i in dialog.GetSelections()]
				else:
					self.animal_to_include=self.animal_kinds
				dialog.Destroy()
			else:
				self.animal_to_include=self.animal_kinds

			dialog=wx.MultiChoiceDialog(self,message='Select behaviors',caption='Behaviors to annotate',choices=list(self.behaviornames_and_colors.keys()))
			if dialog.ShowModal()==wx.ID_OK:
				self.behavior_to_include=[list(self.behaviornames_and_colors.keys())[i] for i in dialog.GetSelections()]
			else:
				self.behavior_to_include=list(self.behaviornames_and_colors.keys())
			dialog.Destroy()

			if len(self.behavior_to_include)==0:
				self.behavior_to_include=list(self.behaviornames_and_colors.keys())
			if self.behavior_to_include[0]=='all':
				self.behavior_to_include=list(self.behaviornames_and_colors.keys())

			if self.behavior_mode==2:
				dialog=wx.MessageDialog(self,'Specify individual-specific behaviors? e.g., sex-specific behaviors only occur in a specific sex and\ncan be used to maintain the correct ID of this individual during the entire analysis.','Individual-specific behaviors?',wx.YES_NO|wx.ICON_QUESTION)
				if dialog.ShowModal()==wx.ID_YES:
					for animal_name in self.animal_kinds:
						dialog1=wx.MultiChoiceDialog(self,message='Select individual-specific behaviors for '+str(animal_name),caption='Individual-specific behaviors for '+str(animal_name),choices=self.behavior_to_include)
						if dialog1.ShowModal()==wx.ID_OK:
							self.specific_behaviors[animal_name]={}
							self.correct_ID=True
							specific_behaviors=[self.behavior_to_include[i] for i in dialog1.GetSelections()]
							for specific_behavior in specific_behaviors:
								self.specific_behaviors[animal_name][specific_behavior]=None
						dialog1.Destroy()
				else:
					self.correct_ID=False
				dialog.Destroy()

			complete_colors=list(mpl.colors.cnames.values())
			colors=[]
			for c in complete_colors:
				colors.append(['#ffffff',c])
			
			dialog=wx.MessageDialog(self,'Specify the color to represent\nthe behaviors in annotations and plots?','Specify colors for behaviors?',wx.YES_NO|wx.ICON_QUESTION)
			if dialog.ShowModal()==wx.ID_YES:
				names_colors={}
				n=0
				while n<len(self.behavior_to_include):
					dialog2=ColorPicker(self,'Color for '+self.behavior_to_include[n],[self.behavior_to_include[n],colors[n]])
					if dialog2.ShowModal()==wx.ID_OK:
						(r,b,g,_)=dialog2.color_picker.GetColour()
						new_color='#%02x%02x%02x'%(r,b,g)
						self.behaviornames_and_colors[self.behavior_to_include[n]]=['#ffffff',new_color]
						names_colors[self.behavior_to_include[n]]=new_color
					else:
						if n<len(colors):
							names_colors[self.behavior_to_include[n]]=colors[n][1]
							self.behaviornames_and_colors[self.behavior_to_include[n]]=colors[n]
					dialog2.Destroy()
					n+=1
				if self.correct_ID:
					self.text_selectbehaviors.SetLabel('Selected: '+str(list(names_colors.keys()))+'. Specific behaviors: '+str(self.specific_behaviors)+'.')
				else:
					self.text_selectbehaviors.SetLabel('Selected: '+str(list(names_colors.keys()))+'.')
			else:
				for color in colors:
					index=colors.index(color)
					if index<len(self.behavior_to_include):
						behavior_name=list(self.behaviornames_and_colors.keys())[index]
						self.behaviornames_and_colors[behavior_name]=color
				if self.correct_ID:
					self.text_selectbehaviors.SetLabel('Selected: '+str(self.behavior_to_include)+' with default colors. Specific behaviors:'+str(self.specific_behaviors)+'.')
				else:
					self.text_selectbehaviors.SetLabel('Selected: '+str(self.behavior_to_include)+' with default colors.')
			dialog.Destroy()

			if self.behavior_mode!=3:
				dialog=wx.MessageDialog(self,'Show legend of behavior names in the annotated video?','Legend in video?',wx.YES_NO|wx.ICON_QUESTION)
				if dialog.ShowModal()==wx.ID_YES:
					self.show_legend=True
				else:
					self.show_legend=False
				dialog.Destroy()


	def select_parameters(self,event):

		if self.behavior_mode>=3:

			wx.MessageBox('No need to specify this since the selected behavior mode is "Static images".','Error',wx.OK|wx.ICON_ERROR)

		else:

			if self.path_to_categorizer is None:
				parameters=['3 areal parameters','3 length parameters','4 locomotion parameters']
			else:
				if self.behavior_mode==1:
					parameters=['count','duration','latency']
				else:
					parameters=['count','duration','latency','3 areal parameters','3 length parameters','4 locomotion parameters']

			dialog=wx.MultiChoiceDialog(self,message='Select quantitative measurements',caption='Quantitative measurements',choices=parameters)
			if dialog.ShowModal()==wx.ID_OK:
				self.parameter_to_analyze=[parameters[i] for i in dialog.GetSelections()]
			else:
				self.parameter_to_analyze=[]
			dialog.Destroy()

			if len(self.parameter_to_analyze)<=0:
				self.parameter_to_analyze=[]
				self.normalize_distance=False
				self.text_selectparameters.SetLabel('NO parameter selected.')
			else:
				if '4 locomotion parameters' in self.parameter_to_analyze:
					dialog=wx.MessageDialog(self,'Normalize the distances by the size of an animal? If no, all distances will be output in pixels.','Normalize the distances?',wx.YES_NO|wx.ICON_QUESTION)
					if dialog.ShowModal()==wx.ID_YES:
						self.normalize_distance=True
						self.text_selectparameters.SetLabel('Selected: '+str(self.parameter_to_analyze)+'; with normalization of distance.')
					else:
						self.normalize_distance=False
						self.text_selectparameters.SetLabel('Selected: '+str(self.parameter_to_analyze)+'; NO normalization of distance.')
					dialog.Destroy()
				else:
					self.normalize_distance=False
					self.text_selectparameters.SetLabel('Selected: '+str(self.parameter_to_analyze)+'.')

	#END COPY

	def select_num_workers(self, event):
		dialog = wx.Dialog(self, title="Set Number of Worker Processes", size=(350, 150))

		vbox = wx.BoxSizer(wx.VERTICAL)

		num_cores = multiprocessing.cpu_count()
		st = wx.StaticText(dialog, label=f"Select number of worker processes (max {num_cores}):")
		vbox.Add(st, 0, wx.ALL | wx.EXPAND, 10)
		self.spin_workers = wx.SpinCtrl(dialog, value="", min=1, max=num_cores)
		vbox.Add(self.spin_workers, 0, wx.ALL | wx.EXPAND, 10)

		btns = dialog.CreateButtonSizer(wx.OK | wx.CANCEL)
		vbox.Add(btns, 0, wx.ALL | wx.ALIGN_CENTER, 10)

		dialog.SetSizer(vbox)

		if dialog.ShowModal() == wx.ID_OK:
			self.num_workers = self.spin_workers.GetValue()
			wx.MessageBox(f"Number of worker processes set to: {self.num_workers}", "Confirmed", wx.OK | wx.ICON_INFORMATION)
		else:
			self.num_workers = 1  # fallback default

		self.text_num_workers.SetLabel("Number of workers: " + str(self.num_workers))

		dialog.Destroy()

	def select_outputpath(self,event):

		dialog=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.out_path=dialog.GetPath()
			self.result_path = self.out_path
			self.text_outputfolder.SetLabel('The output will be in: '+self.out_path+'.')
		dialog.Destroy()

	def input_analysis_interval(self, event):
		dialog = wx.MessageDialog(
			self, 
			'Would you like to set a custom analysis interval?', 
			'Set Analysis Interval?', 
			wx.YES_NO | wx.ICON_QUESTION
		)
		
		if dialog.ShowModal() == wx.ID_YES:
			dialog.Destroy()
			dialog1 = wx.TextEntryDialog(self, 'Enter the custom analysis interval (integer representing milliseconds):', 'Set Analysis Interval')
			
			if dialog1.ShowModal() == wx.ID_OK:
				try:
					self.analysis_interval = int(dialog1.GetValue())
					self.text_analysis_interval.SetLabel(f'Analysis interval set to {self.analysis_interval}ms.')
				except ValueError:
					wx.MessageBox('Invalid input! Please enter an integer.', 'Error', wx.OK | wx.ICON_ERROR)
					self.analysis_interval = 5000  # Reset to default
					self.text_analysis_interval.SetLabel('Invalid input. Default analysis interval = 5000 (milliseconds).')
			
			dialog1.Destroy()
		else:
			self.analysis_interval = 5000  # Default value
			self.text_analysis_interval.SetLabel('Default analysis interval = 5000 (milliseconds).')
		
		dialog.Destroy()

	def save_clip(self, frames, filename, fps):
		"""Saves a list of frames as a video file."""
		height, width, _ = frames[0].shape
		fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4
		out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

		for frame in frames:
			out.write(frame)

		out.release()

		print(f"Saved clip: {filename}")

	def start_analysis(self, status):
		if self.video_input_mode == "camera":
			capture = cv2.VideoCapture(self.camera_source)
		elif self.video_input_mode == "file":
			capture = cv2.VideoCapture(self.path_to_clip)

		if not capture.isOpened():
			print("Error: Could not open camera source.")
			exit(1)
			
		FPS = capture.get(cv2.CAP_PROP_FPS)  # Grab the fps for capturing
		if self.analysis_interval is None:
			self.analysis_interval = 5000

		cv2.namedWindow("Live Stream", cv2.WINDOW_NORMAL)

		clip_number = 0

		# Process safe ordered dictionary
		# Since dictionaries are persistent and python uses references,
		# you can share the memory between threads
		# Check MapReduce project for references
		manager = multiprocessing.Manager()
		status = manager.dict()

		# status["model"] = model
		# status["encodings"] = encodings
		status["analysis_queue"] = manager.Queue()

		# To make sure that the clips don't get added to master out of order
		current_clip_number = manager.Value("i", 0)
		shutdown = manager.Value("i", 0)

		# Just replace with threads to go back to multi-threaded
		# Issue with not enough sleep during behavior analysis, need to swap to processes
		# Uses cores in parallel instead of just taking advantage of I/O waiting
		num_analysis_processes = self.num_workers
		worker_process_list = []

		for i in range(num_analysis_processes):
			worker = Live_Worker(status, self.get_live_config(), current_clip_number, i, shutdown)
			worker_process = multiprocessing.Process(target=worker)
			worker_process_list.append(worker_process)
			worker_process.start()

		running = True

		output_dir = os.path.join(self.out_path, "captured_clips")  # Directory to save clips
		os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
		
		start_time = time.time()
		queue_size_data = []

		while running:
			time.sleep(0.1)
			queue_size_data.append((time.time()-start_time, status['analysis_queue'].qsize(), clip_number))
			print(f"\nCapturing {self.analysis_interval}ms clip {clip_number}...")

			frames = []
			target_frame_count = int(FPS * (self.analysis_interval / 1000))

			for i in range(target_frame_count):
				ret, frame = capture.read()
				if not ret:
					print("End of video reached!")
					running = False
					break

				frames.append(frame)
				cv2.imshow("Live Stream", frame)

				# Exit early if 'q' is pressed
				if cv2.waitKey(1) & 0xFF == ord('q'):
					running = False
					break

			if frames:
				# Analyze behaviors takes a folder not a file, so every different clip needs its own folder ugggggg
				# Save the captured clip
				clip_folder = os.path.join(output_dir, f"clip_{clip_number}")
				os.makedirs(clip_folder, exist_ok=True)  # Create folder for each clip
				clip_filename = os.path.join(clip_folder, f"clip_{clip_number}.mp4")
				self.save_clip(frames, clip_filename, FPS)

				task = {"clip_filename": clip_filename, "clip_folder": clip_folder, "clip_number": clip_number, "creation_time": time.time()}
				status["analysis_queue"].put(task)
				clip_number += 1  # Increment clip count
				print("current_clip_num:", current_clip_number.value)
		
		while not clip_number == current_clip_number.value:
			# Recording is done
			# Wait for the rest of the clips to finish processing
			queue_size_data.append((time.time()-start_time, status['analysis_queue'].qsize(), clip_number))
			time.sleep(5)

		shutdown.value = 1

		# print("Set shutdown value=1")

		# Release resources
		for worker_process in worker_process_list:
			worker_process.join()
		
		# print("Workers joined")

		capture.release()
		cv2.destroyAllWindows()

		times, queue_sizes, clips = zip(*queue_size_data)
		plt.figure()
		plt.plot(times, queue_sizes, marker='o')
		for t, q, c in zip(times, queue_sizes, clips):
			plt.annotate(str(c), (t, q), textcoords="offset points", xytext=(5, 5), ha='left', fontsize=8)

		plt.title("Analysis Queue Size vs Time (Annotated with total clips at time)")
		plt.xlabel("Time Elapsed (s)")
		plt.ylabel("Queue Size")
		plt.grid(True)
		plt.tight_layout()
		plt.savefig(os.path.join(self.out_path, "queue_size_plot.png"))
	
	def get_video_sources(self, max_index=10):
		"""
		Scans for available camera indices and returns a list of strings like '0 (Camera 0)'.
		"""
		sources = []
		# Choose appropriate OpenCV backend based on OS
		system = platform.system()
		if system == 'Darwin':  # macOS
			user_capture = cv2.CAP_AVFOUNDATION
		elif system == 'Windows': # Windows, duh
			user_capture = cv2.CAP_DSHOW
		else:
			user_capture = cv2.CAP_V4L2  # Linux default
		for index in range(max_index):
			cap = cv2.VideoCapture(index, user_capture)
			if cap is not None and cap.read()[0]:
				sources.append(f"{index} (Camera {index})")
			cap.release()
		return sources

	def preview_camera(self, cam_index):
		cap = cv2.VideoCapture(cam_index)

		if not cap.isOpened():
			wx.MessageBox(f"Could not open camera {cam_index}", "Error", wx.OK | wx.ICON_ERROR)
			return

		wx.MessageBox("Press 'q' in the preview window to close the preview.", "Instructions", wx.OK | wx.ICON_INFORMATION)

		while True:
			ret, frame = cap.read()
			if not ret:
				break
			cv2.imshow(f"Camera {cam_index} Preview", frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		cap.release()
		cv2.destroyAllWindows()

	def select_videosource(self, event):
		camera_options = self.get_video_sources()

		if not camera_options:
			wx.MessageBox('No cameras found.', 'Error', wx.OK | wx.ICON_ERROR)
			return

		dialog = wx.SingleChoiceDialog(
			self,
			message='Select a camera to preview:',
			caption='Camera Selection',
			choices=camera_options
		)

		if dialog.ShowModal() == wx.ID_OK:
			selection = dialog.GetStringSelection()
			cam_index = int(selection.split()[0])
			self.camera_source = cam_index
			self.text_videosource.SetLabel(f"Selected camera: {self.camera_source}")
			self.preview_camera(cam_index)  # Open preview on selection
			self.camera_source = cam_index

		dialog.Destroy()
		
	# To pass LabGym analysis config to worker processes
	def get_live_config(self):
		return {
			"out_path": self.out_path,
			"analysis_interval": self.analysis_interval,
			"behavior_mode": self.behavior_mode,
			"use_detector": self.use_detector,
			"detector_path": self.detector_path,
			"path_to_detector": self.path_to_detector,
			"detector_batch": self.detector_batch,
			"detection_threshold": self.detection_threshold,
			"animal_kinds": self.animal_kinds,
			"background_path": self.background_path,
			"model_path": self.model_path,
			"path_to_categorizer": self.path_to_categorizer,
			"framewidth": self.framewidth,
			"delta": self.delta,
			"decode_animalnumber": self.decode_animalnumber,
			"animal_number": self.animal_number,
			"autofind_t": self.autofind_t,
			"decode_t": self.decode_t,
			"t": self.t,
			"duration": self.duration,
			"decode_extraction": self.decode_extraction,
			"ex_start": self.ex_start,
			"ex_end": self.ex_end,
			"behaviornames_and_colors": self.behaviornames_and_colors,
			"dim_tconv": self.dim_tconv,
			"dim_conv": self.dim_conv,
			"channel": self.channel,
			"length": self.length,
			"animal_vs_bg": self.animal_vs_bg,
			"stable_illumination": self.stable_illumination,
			"animation_analyzer": self.animation_analyzer,
			"animal_to_include": self.animal_to_include,
			"behavior_to_include": self.behavior_to_include,
			"parameter_to_analyze": self.parameter_to_analyze,
			"include_bodyparts": self.include_bodyparts,
			"std": self.std,
			"uncertain": self.uncertain,
			"min_length": self.min_length,
			"show_legend": self.show_legend,
			"background_free": self.background_free,
			"black_background": self.black_background,
			"normalize_distance": self.normalize_distance,
			"social_distance": self.social_distance,
			"specific_behaviors": self.specific_behaviors,
			"correct_ID": self.correct_ID,
		}

