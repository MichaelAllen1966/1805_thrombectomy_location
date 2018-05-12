# -*- coding: utf-8 -*-
"""
Created on Wed May 31 17:28:34 2017

@author: kp331
"""

import numpy as np
import pandas as pd
import random as rn
import pyf_ga05_functions_171108 as ga
import datetime
from scipy.spatial.distance import pdist
import os
import math

print ('start', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Set output location

OUTPUT_LOCATION='output/new24_csc_0_1_3_4'
# Create new folder if folder does not already exist
if not os.path.exists(OUTPUT_LOCATION):
    os.makedirs(OUTPUT_LOCATION)

## Directories needed
# Data required to run algorith to be stored in 'data' folder as subfolder of where code is stroes
# 'output' folder required to store output

## Defining scores and which are used in Pareto selection
# Score_matrix:
# 0: Number of hospitals
# 1: Average distance
# 2: Maximum distance
# 3: Minimise Maximum thrombolysis admissions to any one hopsital
# 4: Maximise Minimum thrombolysis admissions to any one hopsital
# 5: Max/Min Admissions ratio
# 6: Proportion thrombolysis patients within target distance 1 (thrombolysis)
# 7: Proportion thrombolysis patients within target distance 2 (thrombolysis)
# 8: Proportion thrombolysis patients within target distance 3 (thrombolysis)
# 9: Proportion thrombolysis patients attending unit with target admission numbers
# 10: Proportion of thrombolysis patients meeting distance 1 and admissions target
# 11: Proportion of thrombolysis patients meeting distance 2 and admissions target
# 12: Proportion of thrombolysis patients meeting distance 3 and admissions target
# 13: Clinical benefit if thrombolysis (fixed door to needle = mins + fixed onset to travelling in ambulance time = mins + travel time which is model dependant).  Additional benefit per 100 treatable patients
# 14: Proportion of patients receiving thrombolysis within target time
# 15: Proportion of patients receiving thrombectomy within target time
# 16: Proportion patients within target distance 1 (thrombectomy: include the direct travel and if necessary the transfer travel to get to CSC which includes the 60 mins that's been included in the inter travel matrix)
# 17: Proportion patients within target distance 2 (thrombectomy: include the direct travel and if necessary the transfer travel to get to CSC which includes the 60 mins that's been included in the inter travel matrix)
# 18: Proportion patients within target distance 3 (thrombectomy: include the direct travel and if necessary the transfer travel to get to CSC which includes the 60 mins that's been included in the inter travel matrix)
# 19: Proportion of CSC that have > TARGET THROMBECTOMY ADMISSIONS
# 20: Minimise the Maximum thrmbectomy admissions to any one hopsital
# 21: Maximise the Minimum thrombectomy admissions to any one hopsital
                                       
#MODELs (set ALLOWABLE_THROMBOLYSIS & ALLOWABLE_THROMBECTOMY_DELAY [only one of these variables can have a value other than 0])
#1)	Patients go to closest unit (set ALLOWABLE_THROMBOLYSIS=0 & ALLOWABLE_THROMBECTOMY_DELAY=0)
#2)	Patients go to CSC if within 45 minutes (set ALLOWABLE_THROMBOLYSIS=-45 & ALLOWABLE_THROMBECTOMY_DELAY=0)
#3)	Patients go to closest unit, but adjust travel time so 15 mins in favour of CSC  (set ALLOWABLE_THROMBOLYSIS_DELAY=15 & ALLOWABLE_THROMBECTOMY_DELAY=0)
#4)	Patients go to HASU if it doesn’t increase time to CSC by 20 minutes (set ALLOWABLE_THROMBOLYSIS=0 & ALLOWABLE_THROMBECTOMY_DELAY=20)

#OPTIONS 
#a)Patients go to closest CSC from HASU
#b)Patients go to closest CSC from home [NOT YET CODED]

#Key Performance indicators
#i)	Patients receive thrombolysis within x minutes (use same door-to-needle for all centres)
#ii)	Patients receive thrombectomy within x minutes (use same door-to-groin & transfer within HASU for all centres)
#iii) Admissions <2000, but not essential, if a mega centre then report that they need more beds
#iv) Admissions >1000, but allow >600 in certain areas
    
pareto_include=np.array([0,1,3,4]) # scores to use in defining pareto front
nscore_parameters=23
rn.seed() # selects random number seed based on clock, Use fixed value in argument for fixed random number sequences
n_combinations=0

## Import data
LOAD_INITAL_POPULATION=0 # Set to 1 to load an initial population. This will be added to any randomly generated population
if LOAD_INITAL_POPULATION==1: 
#    LOAD_POPULATION=np.loadtxt('data/Load_170928_model3_15_mins_24NSCasCSC.csv',delimiter=',')
#    LOAD_POPULATION=np.loadtxt('data/171019_LOAD_model1_28CSC.csv',delimiter=',')#GA model found this to be the largest number of hu
    LOAD_POPULATION=np.loadtxt('data/load.csv',delimiter=',')#Phils list of 27 CSCs 
#HOSPITALS=pd.read_csv('data/hospitals_fix28CSCs_no_Tony_barred.csv') # The GA found this to be the lasrgest nubmer of centres whilst aving >200 MT/centre
HOSPITALS=pd.read_csv('data/hospitals_fix24CSCsasNewNSCs.csv') # All the centres open, as a CSC if a NSC, else as a HASU
HOSPITALS=HOSPITALS.fillna(0)#fill any empty values with 0 (specifically needed, for the 9th column)
HOSPITAL_COUNT=len(HOSPITALS.index)
DO_FIX_HOSPITAL_STATUS=1 # Set to 1 to force a set of hospitals open. File contains a 1 for the hospital position to keep open or closed
a=np.array(HOSPITALS)#Convert panda to np.array
HOSPITAL_STATUS=np.array(a[:,9]).reshape((1,HOSPITAL_COUNT))#take the 5th column from the file and make it a single row
#KP: instead of needing to put "0" in the empty cells for column 9, need to work out how to convert the nan in hospital.csv to zeros. np.isnan and np.nan_to_num do not work

# PATIENT_NODES=pd.read_csv('data/msoa_postcodes_truncated.csv') # List of postcodes (not currently used)
ADMISSIONS=np.loadtxt('data/LSOA_strokes_no_index.csv',delimiter=',') # Admissions for each patient node.  PREVIOUSLY using "msoa_truncated_stemi.csv"
TOTAL_ADMISSIONS=np.sum(ADMISSIONS)
TRAVEL_MATRIX=np.loadtxt('data/stroke_time_matrix_data_only.csv',delimiter=',') # Node to hospital matrix.  PREVIOUSLY using "msoa_truncated_dist_matrix.csv"
TRAVEL_MATRIX_BETWEEN_ALL_HOSPITALS=np.loadtxt('data/inter_hospital_time_plus60_no_index.csv',delimiter=',')#inter_hospital_time_no_index.csv',delimiter=',') # Node to hospital matrix.  PREVIOUSLY using "msoa_truncated_dist_matrix.csv"

#Only ever need the columns for the CSC hospitals, and these are fixed for the entire run
mask_csc_from_all_hospitals=HOSPITAL_STATUS==2#For the number of hospitals in the solution, a T/F for if they are a CSC
mask_csc_from_all_hospitals=np.array(mask_csc_from_all_hospitals).reshape(-1,)
TRAVEL_MATRIX_BETWEEN_ALL_HOSPITALS_AND_CSC=TRAVEL_MATRIX_BETWEEN_ALL_HOSPITALS[:,mask_csc_from_all_hospitals]

#But need to keep all the colu,ns in place otherwise the ID of the csc will be out of line when return the .argmin
#TRAVEL_MATRIX_BETWEEN_ALL_HOSPITALS_9999HASU=ga.hasu_add_travel(len(TRAVEL_MATRIX_BETWEEN_ALL_HOSPITALS),HOSPITAL_STATUS,9999)
                                  
                                  
## Initialise variables
GENERATIONS=500
INITIAL_RANDOM_POPULATION_SIZE=10000
NEW_RANDOM_POPULATION_EACH_GENERATION=0.05 # number of new random mmers each generation as proportion of selected population
MINIMUM_SELECTED_POPULATION_SIZE=1000 
MAXIMUM_SELECTED_POPULATION_SIZE=10000
MUTATION_RATE=0.002 # prob of each gene mutationg each generation
TARGET_THROMBOLYSIS_ADMISSIONS=600#(was 1000 until 17102017)
TARGET_THROMBECTOMY_ADMISSIONS=150#(was 145 untul 17102017, 200 for runs in Oct and Nov.  Using 150 for the National Thrombectomy Paper).  WANT ALL CSC TO HAVE MORE THAN THIS NUMBER OF ADMISSIONS FOR THROMBECTOMY
TARGET_THROMBOLYSIS_TIME=100
TARGET_THROMBECTOMY_TIME=200
SKIP_BREEDING_IN_FIRST_GENERATION=0 # Used to just select pareto front from initial population (without breeding first)
population=np.zeros((0,HOSPITAL_COUNT))
CROSSOVERPROB = 100#% likely to happen
USE_CROWDING=0 #else use thinning
CROSSOVERUNIFORM = False#each gene has equal chance coming from each parent
if CROSSOVERUNIFORM == False:#use standard cross over method, define number of cross over locations
    MAXCROSSOVERPOINTS=3
CALCULATE_HAMMING=True
PROPORTION_ELIGIBLE_THROMBECTOMY=0.1#10% of thrombolysis patients are eligible for thrombectomy


#USER DEFINED PARAMETERS FOR SCORE TARGETS
THROMBOLYSIS_TARGET_DISTANCE_1=30 # straight line km, equivalent to 30 min
THROMBOLYSIS_TARGET_DISTANCE_2=45 # straight line km, equivalent to 45 min
THROMBOLYSIS_TARGET_DISTANCE_3=60 # straight line km, equivalent to 60 min
THROMBECTOMY_TARGET_DISTANCE_1=45 # straight line km, equivalent to 30 min
THROMBECTOMY_TARGET_DISTANCE_2=60 # straight line km, equivalent to 45 min
THROMBECTOMY_TARGET_DISTANCE_3=75 # straight line km, equivalent to 60 min
    
#USER DEFINED PARAMETERS FOR PROCESS TIMES
DOOR_TO_NEEDLE = 58#all hospoital have same door to needle taken as national average SSNAP April 2015-March 2016
INHOSPITAL_TRANSFER = 5#nominal time to move from thrombolysis to thrombectomy (if already in CSC), or from HASU to ambulance
DOOR_TO_GROINdir = 60#all hospitals have same door to groin (for patients already in CSC for their thrombolysis)
#    DOOR_TO_GROINindir = 60#all hospoital have same door to groin(for patients transfered from HASUS to CSC for their thrombectomy)

#SENSOR CHECK THE POPULATION VALUES AS CAN ONLY ASK FOR A SIZE THAT'S < THE NUMBER OF POSSIBLE OPTIONS, OTHERWISE GET STUCK IN A LOOP THAT ONCE REMOVE DUPLICATE SOLUTIONS, CAN NOT CREATE ENOUGH SOLUTIONS!
nHOSPITALS_FREE_CHOICE=HOSPITAL_COUNT-(np.count_nonzero(HOSPITAL_STATUS==1)+np.count_nonzero(HOSPITAL_STATUS==2)+np.count_nonzero(HOSPITAL_STATUS==-1))#Hospitals can choose are those without a fixed status as set in the input file

#From n obects (nHOSPITALS_FREE_CHOICE), take r in a sample
for r in range(nHOSPITALS_FREE_CHOICE):
    n_combinations=n_combinations+((math.factorial(nHOSPITALS_FREE_CHOICE))/(math.factorial(r)*math.factorial(nHOSPITALS_FREE_CHOICE-r)))

#Check the parameters are sensible.  If the minimum requested exceeds the full number of options, set it to be 1/10 of the possibilities.  Do not worry about maximum
if n_combinations<INITIAL_RANDOM_POPULATION_SIZE:
    INITIAL_RANDOM_POPULATION_SIZE=np.int(n_combinations/10)
if n_combinations<MINIMUM_SELECTED_POPULATION_SIZE:
    MINIMUM_SELECTED_POPULATION_SIZE=np.int(n_combinations/10)


#CHOOSE BETWEEN 4 WAYS OF SELECTING WHERE PATIENT GOES FOR THROMBOLYSIS IF CSC CENTRES ARE SET UP

ALLOWABLE_THROMBECTOMY_DELAY=0#, model 4 (ALLOWABLE_THROMBECTOMY_DELAY = +ve).  OVERWRITES ALLOWANCE, AS ONE OR THE OTHER
if ALLOWABLE_THROMBECTOMY_DELAY==0:
    ALLOWABLE_THROMBOLYSIS_DELAY = 0#the extra time for a patient to travel to a CSC over a nearer HASU
    #ALLOWABLE_THROMBOLYSIS_DELAY is used to indicate whether it's model 1 (ALLOWABLE_THROMBOLYSIS_DELAY=0), model 2 (ALLOWABLE_THROMBOLYSIS_DELAY = -ve [if within distance then go to CSC]), model 3 (ALLOWABLE_THROMBOLYSIS_DELAY = +ve [CSC has larger catchement advantage])
else:
    ALLOWABLE_THROMBOLYSIS_DELAY = 0#model3: the extra time for a patient to travel to a CSC over a nearer HASU
    
#TRAVEL_MATRIX_ADJ=ga.travel_matrix_adjust_csc(TRAVEL_MATRIX,HOSPITAL_STATUS,ALLOWABLE_THROMBOLYSIS_DELAY)
    
#VALUES FOR EPSILON
#Set to 0 for just max, 1 for just min, 2 for min and max, 3 for none
WHICH_EPSILON = 3
#the number of generations with the same epsilon to determine ending the algorithm as it's not improving
CONSTANT_EPSILON=999
#stores the epsilon for the generation
epsilonmin=np.zeros(GENERATIONS)
epsilonmax=np.zeros(GENERATIONS)
#gets set to TRUE if the Epsilon (min, max, both) has been static for more than CONSTANT_EPSILON generations
static_epsilon = False

## Set up score normalisation array
# Set array to normalise all scores 0-1 (pareto needs higher=better).
# Absolute values not critical but high/low is. First number will score zero. Second number will score 1
# There is no truncation of scores
NORM_MATRIX=np.array([[HOSPITAL_COUNT,1],[250,0],[250,0],[TOTAL_ADMISSIONS,1],[1,TOTAL_ADMISSIONS],[25,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,100],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[TOTAL_ADMISSIONS,1],[1,TOTAL_ADMISSIONS],[250,0]])

GA_DIVERSITY_PROGRESS=np.zeros((GENERATIONS,6)) #Store key information per generation to monitor how diverse the population is, and how strong it is
#i=Generation number
#(i,1) Size of first pareto front

#The following are recording a measure of 'diversity'
#(i,2) Hamming distance of the breeding population
#(i,3) Hamming distance of first pareto front

#The following are recording a measure of 'progress'
#(i,4) Max number for % patients attending unit within 30 mins, having 600 admissions [column 10 in score].. kp chose that to be from first Pareto
#(i,5) Epsilon
#(i,6) Hypervolume

## Generate initial random population and add to any loaded population.  Check that enough is created once remove duplicates
population_size=0
FirstTime=0

while population_size<MINIMUM_SELECTED_POPULATION_SIZE: 
## Generate initial random population and add to any loaded population
    if INITIAL_RANDOM_POPULATION_SIZE>0:
        population=np.vstack((population,ga.generate_random_population(INITIAL_RANDOM_POPULATION_SIZE,HOSPITAL_COUNT)))
        
    #Only want to add in the loaded population once.  If after removing duplicates there are not enough population, then only generate more, not reload the same
    if LOAD_INITAL_POPULATION==1 and FirstTime==0:
        population=np.vstack((LOAD_POPULATION,population))

    ### Fix open all necessary hospitals
    if DO_FIX_HOSPITAL_STATUS==1:
     #Takes the 10th column from the hospital.csv file and if "1" then open, "-1" then closed, if "2" then open
         population=ga.fix_hospital_status(population,HOSPITAL_STATUS)

    ### Remove all zero rows, and remove non-unique rows
    check_hospitals=np.sum(population,axis=1)>0 
    population=population[check_hospitals,:]
    population=ga.unique_rows(population)
        
    population_size=len(population[:,0]) # current population size. Number of children generated will match this.
    FirstTime=1
    INITIAL_RANDOM_POPULATION_SIZE=MINIMUM_SELECTED_POPULATION_SIZE-population_size+1000#generate enough to meet the minimum, and some extra (1000) so that once duplicates are removed it will still meet it
 
print('Begin generations: ',datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'Starting population size: ',population_size)
#KP DO: SAVE FIRST GENERATION TO DETECT PROGRESS ON THE LASTEST SAVED PARETO FRONT
## Genetic algorithm. Breed then use pareto selection
for generation in range(GENERATIONS):
    if not static_epsilon: #when Epsilon is static, then stop breeding - the idea is that you've found a good stabel populaiton that will not improve further
    #Being static is dependant on: number of generation choose to need to be the same (CONSTANT_EPSILON), and which epsilon to be static (max: take all the worst attributes for each person, use the best of these. min: take all the worst attributes for each person, use the worst of these. Or need both min and max to be the same)
        #hamming distance of population (before randoms added) and before breeding
#        print('1. MAIN CODE Calculate Hamming: ',datetime.datetime.now().strftime("%H:%M:%S.%f"),' . Generation: ',generation)

        if CALCULATE_HAMMING:
            hamming_distances=pdist(population,'hamming')#the proportion which disagree
            average_hamming=np.average(hamming_distances)
        else:
            average_hamming=999
        GA_DIVERSITY_PROGRESS[generation,2]=average_hamming #"AVERAGE HAMMING DISTANCE, 1=most diverse, 0=homogenous
    
        # Add in random population if required, as fraction of total population (adult+child)
        if NEW_RANDOM_POPULATION_EACH_GENERATION>0:  
            population_size=len(population[:,0])
            new_random_population=ga.generate_random_population(int(NEW_RANDOM_POPULATION_EACH_GENERATION*population_size),HOSPITAL_COUNT)
            ### Fix open or closed all necessary hospitals
            if DO_FIX_HOSPITAL_STATUS==1:
                new_random_population=ga.fix_hospital_status(new_random_population,HOSPITAL_STATUS)

            population=np.vstack((population,new_random_population)) # add random population to current population

        ### Remove all zero rows, and remove non-unique rows
        check_hospitals=np.sum(population,axis=1)>0 
        population=population[check_hospitals,:]
        population=ga.unique_rows(population)

        population_size=len(population[:,0]) # current population size. Number of children generated will match this.

        if generation>0 or SKIP_BREEDING_IN_FIRST_GENERATION!=1: # used to skip breeding in first generation if wanted
            child_population=np.zeros((int(population_size/2)*2,HOSPITAL_COUNT)) #create empty matrix for children (ensure this is an even number)
    #        for mating in range (0,int(population_size/2)*2,2): # each mating will produce two children (the mating count jumps in steps of 2)
            for mating in range (0,int(population_size/2)*2,2): # each mating will produce two children (the mating count jumps in steps of 2)            
                parent1_ID=rn.randint(0,population_size-1) # select parent ID at random
                parent2_ID=rn.randint(0,population_size-1) # select parent ID at random
                parent=np.vstack((population[parent1_ID,:],population[parent2_ID,:])) # chromosome of parent 1 & 2
                crossoveroccur=rn.randint(0,100)#            print("Probability of crossover occuring is", CROSSOVERPROB ,".  This go it's", crossoveroccur,".  If < then CROSSOVER, else NO CROSSOVER AND PARENTS MAKE IT THROUGH TO NEXT GENERATION")
                if crossoveroccur<CROSSOVERPROB:
                    if CROSSOVERUNIFORM == True:  #USING 2 PARENTS,  CREATE CHILD1 AND CHILD2 USING CROSSOVER
                        #UNIFORM CROSSOVER: EACH GENE HAS EQUAL CHANCE TO COME FROM EACH PARENT
                        child=ga.f_uniform_crossover(parent,HOSPITAL_COUNT)                
                    else:
                        #CROSSOVER HAPPENS IN A SET NUMBER OF LOCATIONS (THE MAXIMUM NUMBER OF LOCATION IS USER DEFINED).  MIN=1
                        child=ga.f_location_crossover(parent, MAXCROSSOVERPOINTS,HOSPITAL_COUNT)
                else:
                    #NO CROSSOVER, PARENTS GO TO NEXT GENERATION
                    child=parent
                child_population[mating:mating+2,:]=child
            ### Random mutation
            random_mutation_array=np.random.random(size=(len(child_population[:,0]),HOSPITAL_COUNT)) # random numbers 0-1
            random_mutation_array[random_mutation_array<=MUTATION_RATE]=1 # set all values not to be mutated to 1
            random_mutation_array[random_mutation_array<1]=0 # set all other values to zero
            child_population=(1-random_mutation_array)*child_population+(random_mutation_array*(1-child_population)) # Inverts zero or one when mutation array value is one
            
            ### Fix open or closed all necessary hospitals
            if DO_FIX_HOSPITAL_STATUS==1:
                child_population=ga.fix_hospital_status(child_population,HOSPITAL_STATUS)
      
            # Add child population to adult population    
            population=np.vstack((population,child_population))    
    
        ### Remove all zero rows, and remove non-unique rows
        check_hospitals=np.sum(population,axis=1)>0 
        population=population[check_hospitals,:]
        population=ga.unique_rows(population)
        # np.savetxt(OUTPUT_LOCATION+'/whole_population_after_breeding_and_randoms.csv',population,delimiter=',',newline='\n')
    
      #  population_size=len(population[:,0])
    
        # Select pareto front
        unselected_population=population
        population=np.zeros((0,HOSPITAL_COUNT))
        population_size=0
        
        #kp: NEED TO REMOVE THE FINAL MATRIX BEING PASSED IN AND CHANGE CODE IN GA.SCORES TO USE THE TRIMMED "TRAVEL_MATRIX_BETWEEN_ALL_HOSPITALS_AND_CSC".  LEFT BOTH IN AS A SAFETY MEASURE WHILE MAKING THE OTHER WORK
        (unselected_scores,thrombolysis_admissions_matrix,transferred_admissions_matrix,thrombectomy_admissions_matrix,total_patients_matrix)=ga.score(unselected_population,TARGET_THROMBOLYSIS_ADMISSIONS,TRAVEL_MATRIX,ADMISSIONS,TOTAL_ADMISSIONS,pareto_include,generation==GENERATIONS-1,nscore_parameters,HOSPITAL_STATUS,ALLOWABLE_THROMBOLYSIS_DELAY,ALLOWABLE_THROMBECTOMY_DELAY,TRAVEL_MATRIX_BETWEEN_ALL_HOSPITALS_AND_CSC,TARGET_THROMBOLYSIS_TIME,TARGET_THROMBECTOMY_TIME,TARGET_THROMBECTOMY_ADMISSIONS,PROPORTION_ELIGIBLE_THROMBECTOMY,THROMBOLYSIS_TARGET_DISTANCE_1,THROMBOLYSIS_TARGET_DISTANCE_2,THROMBOLYSIS_TARGET_DISTANCE_3,THROMBECTOMY_TARGET_DISTANCE_1,THROMBECTOMY_TARGET_DISTANCE_2,THROMBECTOMY_TARGET_DISTANCE_3,DOOR_TO_NEEDLE,INHOSPITAL_TRANSFER,DOOR_TO_GROINdir)
        
        GA_DIVERSITY_PROGRESS[generation,4]=np.amax(unselected_scores[:,10]) #The max number for % patients attending unit within 30 mins, having 600 admissions [column 10 in score].. kp chose that to be from first Pareto
        scores=np.zeros((0,nscore_parameters))   
        store_pareto = True
        
        # The following either 
        # 1) adds more members, from successively lower pareto fronts (from the remaining unselected population), if population size not large enough
        # Or 2) if population size exceeds the MAXIMUM_SELECTED_POPULATION_SIZE then solutions are pruned using crowding_selection (choosing members with the largest crowding distance) or thinning (at random)
        
    # Keep adding Pareto fronts until minimum population size is met
    
    #    #hamming distance of whole population before pareto (after breeding)
    #    hamming_distances=pdist(unselected_population,'hamming')#the proportion which disagree
    #    GA_DIVERSITY_PROGRESS[generation,2]=np.average(hamming_distances) #"AVERAGE HAMMING DISTANCE, 1=most diverse, 0=homogenous
    
        while population_size<MINIMUM_SELECTED_POPULATION_SIZE: 
            max_new_population=MAXIMUM_SELECTED_POPULATION_SIZE-population_size # new maximum population size to add
            norm_unselected_scores=ga.normalise_score(unselected_scores,NORM_MATRIX) # Set array to normalise all scores 0-1 (pareto needs higher=better)
    #        norm_scores=ga.normalise_score(unselected_scores,NORM_MATRIX) # normalise scores 
            score_matrix_for_pareto=norm_unselected_scores[:,pareto_include] # select scores to use in Pareto selection
            #np.savetxt(OUTPUT_LOCATION+'/pre_pareto_scores.csv',score_matrix_for_pareto,delimiter=',',newline='\n')
            pareto_index=ga.pareto(score_matrix_for_pareto) # Pareto selection
            new_pareto_front_population=unselected_population[pareto_index,:] # New Pareto population
            new_pareto_front_scores=unselected_scores[pareto_index,:] # New Pareto population scores
            np.savetxt(OUTPUT_LOCATION+'/pareto_scores.csv',new_pareto_front_scores,delimiter=',',newline='\n')
    
            if store_pareto:
                # Save the first Pareto front (lower, or pruned, Pareto Fronts may be used in the next breeding population but are not stored here)
                # np.savetxt(OUTPUT_LOCATION+'/scores.csv',new_pareto_front_scores,delimiter=',',newline='\n')
                # np.savetxt(OUTPUT_LOCATION+'/hospitals.csv',new_pareto_front_population,delimiter=',',newline='\n')
                scores_hospitals=np.hstack((new_pareto_front_scores,new_pareto_front_population))
                np.savetxt(OUTPUT_LOCATION+'/scores_hospitals.csv',scores_hospitals,delimiter=',',newline='\n')
                new_pareto_front_thrombolysis_admissions=thrombolysis_admissions_matrix[pareto_index,:] # New Pareto population scores
                np.savetxt(OUTPUT_LOCATION+'/thrombolysis_admissions.csv',new_pareto_front_thrombolysis_admissions,delimiter=',',newline='\n')
                new_pareto_front_transferred_admissions=transferred_admissions_matrix[pareto_index,:] # New Pareto population scores
                np.savetxt(OUTPUT_LOCATION+'/transferred_admissions.csv',new_pareto_front_transferred_admissions,delimiter=',',newline='\n')
                new_pareto_front_thrombectomy_admissions=thrombectomy_admissions_matrix[pareto_index,:] # New Pareto population scores
                np.savetxt(OUTPUT_LOCATION+'/thrombectomy_admissions.csv',new_pareto_front_thrombectomy_admissions,delimiter=',',newline='\n')
                new_pareto_front_total_patients=total_patients_matrix[pareto_index,:] # New Pareto population scores
                np.savetxt(OUTPUT_LOCATION+'/total_patients.csv',new_pareto_front_total_patients,delimiter=',',newline='\n')
                #np.savetxt(OUTPUT_LOCATION+'/epsilon min.csv',epsilonmin,delimiter=',',newline='\n')
                #np.savetxt(OUTPUT_LOCATION+'/epsilon max.csv',epsilonmax,delimiter=',',newline='\n')
                store_pareto = False
                GA_DIVERSITY_PROGRESS[generation,1]=len(new_pareto_front_population[:,0])
                #hamming distance of first pareto front
                if CALCULATE_HAMMING:
                    hamming_distances=pdist(new_pareto_front_population,'hamming')#the proportion which disagree
                    average_hamming=np.average(hamming_distances)
                else:
                    average_hamming=999
                #"AVERAGE HAMMING DISTANCE, 1=most diverse, 0=homogenous
                GA_DIVERSITY_PROGRESS[generation,3]=average_hamming
                np.savetxt(OUTPUT_LOCATION+'/hamming.csv',GA_DIVERSITY_PROGRESS,delimiter=',',newline='\n')
     
            # Selecting another Pareto front may take population above maximum size, in which latest add population is reduced
            if len(new_pareto_front_population[:,0])>max_new_population:
                # Pick subset.  Either by crowding distance (based on level of clustering of points), or thinning
                if USE_CROWDING==1:#Crowding distance
                    (new_pareto_front_population,new_pareto_front_scores)=ga.crowding_selection(new_pareto_front_population,new_pareto_front_scores,max_new_population)
                else: #use thinning (selecting population to be removed at random so within the right population limit)
                    pick_list=np.zeros(len(new_pareto_front_population[:,0]))            
                    pick_list[0:max_new_population]=1 # Add required number of 1s (to indicate that these are to be removed from the population)
                    np.random.shuffle(pick_list)
                    new_pareto_front_population=new_pareto_front_population[pick_list==1,:]
                    new_pareto_front_scores=new_pareto_front_scores[pick_list==1,:]
      
            else: # Identify remaining unselected population
                unselected_index=np.logical_not(pareto_index)
                unselected_population=unselected_population[unselected_index,:]
                unselected_scores=unselected_scores[unselected_index,:]
            # Add latest selection (population and scores) to previous selection, and remeaure size 
            population=np.vstack((population,new_pareto_front_population))
            scores=np.vstack((scores,new_pareto_front_scores))
            population_size=len(population[:,0])
    
        max_result=np.amax(scores[:,10]) # Use to display progress
        #Calculate Epsilon (measure of how good the population is attaining the reference point (this is all with value 1 after normalising the scores))        
        norm_scores=ga.normalise_score(scores,NORM_MATRIX)

        if WHICH_EPSILON<3:#=3 means don't do an epsilon check
        #When epsilon is static for CONSTANT_EPSILON generations, then static_epsilon is set to True and so will jump over the bulk of the code for the remaining generations
            score_matrix_for_epsilon=norm_scores[:,pareto_include] # select scores to use in Pareto selection
            np.savetxt('output/pre_epsilon_scores.csv',score_matrix_for_epsilon,delimiter=',',newline='\n')
            (epsilonmin[generation],epsilonmax[generation])=ga.calculate_epsilon(score_matrix_for_epsilon)#(nscore_parameters,score_matrix_for_epsilon,population_size)
            if generation>=CONSTANT_EPSILON-1:#only check if there's enough prior generation to go through
                static_epsilon = ga.is_epsilon_static(epsilonmin,epsilonmax,generation,CONSTANT_EPSILON,WHICH_EPSILON)
        
        # At end of generation print time, generation and population size (for monitoring)
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'Generation: ',generation+1,' Population size: ',population_size,' Hamming: ',average_hamming,' Max result: ',max_result)

