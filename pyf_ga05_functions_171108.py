import numpy as np
import random as rn
#import datetime
#import pandas as pd

def calculate_crowding(scores):
#    print('CALCULATE_CROWDING start: ',datetime.datetime.now().strftime("%H:%M:%S.%f"))

    # Crowding is based on chrmosome scores (not chromosome binary values)
    # All scores are normalised between low and high
    # For any one score, all solutions are sorted in order low to high
    # Crowding for chromsome x for that score is the difference between th enext highest and next lowest score
    # Total crowding value sums all crowding for all scores
    population_size=len(scores[:,0])
    number_of_scores=len(scores[0,:])
    # create crowding matrix of population (row) and score (column)
    crowding_matrix=np.zeros((population_size,number_of_scores)) 
    # normalise scores
    normed_scores = (scores-scores.min(0))/scores.ptp(0) # numpy ptp is range (max-min)
    # Calculate crowding
    for col in range(number_of_scores): # calculate crowding distance for each score in turn
        crowding=np.zeros(population_size) # One dimensional array
        crowding[0]=1 # end points have maximum crowding
        crowding[population_size-1]=1 # end points have maximum crowding
        sorted_scores=np.sort(normed_scores[:,col]) # sort scores
        sorted_scores_index=np.argsort(normed_scores[:,col]) # index of sorted scores
        crowding[1:population_size-1]=sorted_scores[2:population_size]-sorted_scores[0:population_size-2] # crowding distance
        re_sort_order=np.argsort(sorted_scores_index) # re-sort to original order step 1
        sorted_crowding=crowding[re_sort_order] # re-sort to orginal order step 2
        crowding_matrix[:,col]=sorted_crowding # record crowding distances
    crowding_distances=np.sum(crowding_matrix,axis=1) # Sum croding distances of all scores
    return crowding_distances
    
#    print('CALCULATE_CROWDING stop: ',datetime.datetime.now().strftime("%H:%M:%S"))
def crowding_selection(population,scores,number_to_select):
    # This function selects a number of solutions based on tournament of crowding distances
    # Two members of the population ar epicked at random
    # The one with the higher croding dostance is always picked
    crowding_distances=calculate_crowding(scores) # crowding distances for each member of the population
    picked_population=np.zeros((number_to_select,len(population[0,:]))) # array of picked solutions (actual solution not ID)
    picked_scores=np.zeros((number_to_select,len(scores[0,:]))) # array of scores for picked solutions
    for i in range(number_to_select):
        population_size=len(population[:,0])
        fighter1ID=rn.randint(0,population_size-1) # 1st random ID
        fighter2ID=rn.randint(0,population_size-1) # 2nd random ID
        if crowding_distances[fighter1ID]>=crowding_distances[fighter2ID]: # 1st solution picked
            picked_population[i,:]=population[fighter1ID,:] # add solution to picked solutions array
            picked_scores[i,:]=scores[fighter1ID,:] # add score to picked solutions array
            # remove selected solution from available solutions
            population=np.delete(population,(fighter1ID), axis=0) # remove picked solution - cannot be chosen again
            scores=np.delete(scores,(fighter1ID), axis=0) # remove picked score (as line above)
            crowding_distances=np.delete(crowding_distances,(fighter1ID), axis=0) # remove crowdong score (as line above)
        else: # solution 2 is better. Code as above for 1st solution winning
            picked_population[i,:]=population[fighter2ID,:]
            picked_scores[i,:]=scores[fighter2ID,:]
            population=np.delete(population,(fighter2ID), axis=0)
            scores=np.delete(scores,(fighter2ID), axis=0)
            crowding_distances=np.delete(crowding_distances,(fighter2ID), axis=0)
    return (picked_population,picked_scores)

def generate_random_population(rows,cols):
    population=np.zeros((rows,cols)) # create array of zeros
    for i in range(rows):
        x=rn.randint(1,cols) # Number of 1s to add
        population[i,0:x]=1 # Add requires 1s
        np.random.shuffle(population[i]) # Shuffle the 1s randomly
    return population
   
def pareto(scores):
    # In this method the array 'scores' is passed to the function.
    # Scores have been normalised so that higher values dominate lower values.
    # The function returns a Boolean array identifying which rows of the array 'scores' are non-dominated (the Pareto front)
    # Method based on assuming everything starts on Pareto front and then records dominated points
    pop_size=len(scores[:,0])
    pareto_front=np.ones(pop_size,dtype=bool)
    for i in range(pop_size):
        for j in range(pop_size):
            if all (scores[j]>=scores[i]) and any (scores[j]>scores[i]):
                # j dominates i
                pareto_front[i]=0
                break
    return pareto_front
    
def normalise_score(score_matrix,norm_matrix):
    # normalise 'score matrix' with reference to 'norm matrix' which gives scores that produce zero or one
    norm_score=np.zeros(np.shape(score_matrix)) # create normlaises score matrix with same dimensions as original scores
    number_of_scores=len(score_matrix[0,:]) # number of different scores
    for col in range(number_of_scores): # normaise for each score in turn
        score_zero=norm_matrix[col,0]
        score_one=norm_matrix[col,1]
        score_range=score_one-score_zero
        norm_score[:,col]=(score_matrix[:,col]-score_zero)/score_range
    return norm_score

def score(population,TARGET_THROMBOLYSIS_ADMISSIONS,FULL_TRAVEL_MATRIX,NODE_ADMISSIONS,TOTAL_ADMISSIONS,pareto_include,CALC_ALL,nscore_parameters,FULL_HOSPITAL_STATUS,ALLOWABLE_THROMBOLYSIS_DELAY,ALLOWABLE_THROMBECTOMY_DELAY,TRAVEL_MATRIX_BETWEEN_ALL_HOSPITALS_AND_CSC,TARGET_THROMBOLYSIS_TIME,TARGET_THROMBECTOMY_TIME,TARGET_THROMBECTOMY_ADMISSIONS,proportion_eligible_thrombectomy,THROMBOLYSIS_TARGET_DISTANCE_1,THROMBOLYSIS_TARGET_DISTANCE_2,THROMBOLYSIS_TARGET_DISTANCE_3,THROMBECTOMY_TARGET_DISTANCE_1,THROMBECTOMY_TARGET_DISTANCE_2,THROMBECTOMY_TARGET_DISTANCE_3,DOOR_TO_NEEDLE,INHOSPITAL_TRANSFER,DOOR_TO_GROINdir):

#    print('SCORE start: ',datetime.datetime.now().strftime("%H:%M:%S.%f"))

#Only calculate the score that is needed by the pareto front, as determined by the array: pareto_include
#Unless CALC_ALL=True (set for the last generation) as then print out all the parameter values

    CALC_ALL=True # MA reporting all

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
# 22: Acerage time to thrombectomy

    
    pop_size=len(population[:,0]) # Count number of solutions to evaluate

#INITIALISE THE ARRAYS FOR STORING RESULTS
    score_matrix=np.zeros((pop_size,nscore_parameters)) # Create an empty score matrix
    thrombolysis_admissions_matrix=np.zeros((pop_size,len(FULL_TRAVEL_MATRIX[0,:])))#store the hospital admissions, col = hospital, row = population
    thrombectomy_admissions_matrix=np.zeros((pop_size,len(FULL_TRAVEL_MATRIX[0,:])))#Where patients go for their thrombectomy. Store the hospital admissions, col = hospital, row = population
    transferred_admissions_matrix=np.zeros((pop_size,len(FULL_TRAVEL_MATRIX[0,:])))#Where patients go for their thrombectomy. Store the hospital admissions, col = hospital, row = population
    total_patients_matrix=np.zeros((pop_size,len(FULL_TRAVEL_MATRIX[0,:])))#Where patients go for their thrombectomy. Store the hospital admissions, col = hospital, row = population


    for i in range(pop_size): # Loop through population of solutions
    
#        print('POPULATION LOOP:  ',i, " ",datetime.datetime.now().strftime("%H:%M:%S.%f"))

        node_results=np.zeros((len(NODE_ADMISSIONS),16))
        # Node results stores results by patient node. These are used in the calculation of results 
        # Node results may be of use to export at later date (e.g. for detailed analysis of one scenario)
        # Col 0: Distance to closest hospital
        # Col 1: Patients within target distance 1 (boolean)
        # Col 2: Patients within target distance 2 (boolean)
        # Col 3: Patients within target distance 3 (boolean)
        # Col 4: Hospital ID
        # Col 5: Number of admissions to hospital ID 
        # Col 6: Does hospital meet admissions target (boolean)
        # Col 7: Admissions and target distance 1 both met (boolean)
        # Col 8: Admissions and target distance 2 both met (boolean)
        # Col 9: Admissions and target distance 3 both met (boolean)
        # Col 10: Time to thrombolysis treatment
        # Col 11: Time to thrombectomy treatment
        # Col 12: Number of thrombectomy admissions to hospital ID 
        # Col 13: Travel time to CSC (so either just direct ot CSC, or the direct ot HASU + transfer travel)

        # Node choice stores by patient node the nearest HASU and CSC and other travel values related to these nearest locations.
        # Then use the user defined values to decide where that patient goes
        # Col 0: Distance to closest solution hospital
        # Col 1: Distance to closest solution CSC
        # Col 2: ID of closest solution hospital
        # Col 3: ID of closest solution CSC
        # Col 4: Distance from HASU to closest CSC from closest solution hosptial
        # Col 5: ID for the nearest solution CSC from the closest solution hospital

#KP REMOVE THIS LINES ONCE GET IT WORKING.... THIS IS JUST A KNOWN CASE TO CHECK IT WORK      
#        population[i] = np.array([0,1,1,0,1,1,0,1,1,0,1,1,0,1])
        # Count hospitals in each solution
        if 0 in pareto_include or CALC_ALL:
            score_matrix[i,0]=np.sum(population[i])

#SET UP THE MASKS AND THEN MASK THE INPUT DATA

        # Initiate matrix of number of admissions to each hospital (first hospital), and to each CSC
        thrombolysis_admissions=np.zeros(np.int(score_matrix[i,0]))
        thrombectomy_admissions=np.zeros(np.int(score_matrix[i,0]))
        transferred_admissions=np.zeros(np.int(score_matrix[i,0]))            

 #       print('A1.',datetime.datetime.now().strftime("%H:%M:%S.%f"))

        # Create the mask to extract from the whole dataset just those hospitals open in the solution
        mask_hospitals_in_solution=np.array(population[i],dtype=bool)#For each hospital in the full list: T/F for those open in the solution
        # hospital_list=np.where(mask) # list of hospitals in selection. Not currently used
        
        # Use this mask to include only the hospital open in the travel data matrix (from patient to hospitals)
        masked_TRAVEL_MATRIX=FULL_TRAVEL_MATRIX[:,mask_hospitals_in_solution]
        masked_HOSPITAL_STATUS=FULL_HOSPITAL_STATUS[:,mask_hospitals_in_solution]


#COLUMNS ARE ONLY EVER THE FIXED CSC, THEN FOR EACH SOLUTION THE ROWS ARE FILTERED DEPENDING ON THE HOSPITALS OPEN IN THE SOLUTION 
        masked_TRAVEL_MATRIX_BETWEEN_SOLUTION_HOSPITALS_AND_CSC=TRAVEL_MATRIX_BETWEEN_ALL_HOSPITALS_AND_CSC[mask_hospitals_in_solution,:]
        #Get CSC id from hospital solution set, from the CSC set
        mask_csc_from_solution_hospitals=masked_HOSPITAL_STATUS==2#For the number of hospitals in the solution, a T/F for if they are a CSC
        mask_csc_from_solution_hospitals=np.array(mask_csc_from_solution_hospitals).reshape(-1,)
        #want an array containing the ID's for the hospitals open in the solution that are CSCs.  So if fromt he 127 full hospital list, a solution has 90 hosptials open and 40 CSCs then the array will have 40 items for ID between 0-90 that are the CSCs
        #pull out the locations for all the "Trues" from the array "mask_csc_from_solution_hosptials"
        csc_id_from_solution_hospitals = np.asarray(np.where(mask_csc_from_solution_hospitals==True))

#Calculate the neareset CSC for each hospital once and use it in the code below
 #       print('A2.',datetime.datetime.now().strftime("%H:%M:%S.%f"))
        TIME_csc_nearest_to_HASU=np.empty(0)
        ID_csc_nearest_to_HASU=np.empty(0)
        for n in range(len(masked_TRAVEL_MATRIX_BETWEEN_SOLUTION_HOSPITALS_AND_CSC)):
            TIME_csc_nearest_to_HASU=np.append(TIME_csc_nearest_to_HASU,np.amin(masked_TRAVEL_MATRIX_BETWEEN_SOLUTION_HOSPITALS_AND_CSC[n,:]))
            ID_csc_nearest_to_HASU=np.append(ID_csc_nearest_to_HASU,csc_id_from_solution_hospitals[0,np.argmin(masked_TRAVEL_MATRIX_BETWEEN_SOLUTION_HOSPITALS_AND_CSC[n,:])]) #store the CSC ID (from the hospital solution list) that has the shortest distance from the first hosptial

#        print('A3.',datetime.datetime.now().strftime("%H:%M:%S.%f"))

        #mask the travel matrix so patients only choose their nearest CSC
        masked_CSC_TRAVEL_MATRIX=masked_TRAVEL_MATRIX[:,mask_csc_from_solution_hospitals]


#Count number of CSC in SOLUTION
#https://stackoverflow.com/questions/28663856/how-to-count-the-occurrence-of-certain-item-in-an-ndarray-in-python
        CSC_COUNT=(masked_HOSPITAL_STATUS == 2).sum() 

#THIS SECTION DETERMINES WHERE TO GO FOR THROMBOLYSIS (BASED ON MODEL 1, 2, 3 or 4)
        # Depending on the model use (1,2,3 or 4) calculate which is each patients first hosptial, and their distance.
        #Important vairalbes returned are:
            #1) first_hospital_masked_ID (this is the ID for the number of hospitals open in the solution)
            #2) node_results[:,0] (the distance to the first_hosptials_masked_ID for each patient)

        if ALLOWABLE_THROMBECTOMY_DELAY>0:
#MODEL4: GO TO HASU FIRST IF DOING SO DOESN'T DELAY GETTING TO CSC BY x MINUTES

   #         print('MODEL4 START',datetime.datetime.now().strftime("%H:%M:%S.%f"))

            node_choice=np.zeros((len(NODE_ADMISSIONS),10))

            #FIND CLOSEST HOSPITAL & CSC FROM THOSE OPEN IN THE SOLUTION
            node_choice[:,0]=np.amin(masked_TRAVEL_MATRIX,axis=1) # distance to closest solution hospital
            node_choice[:,1]=np.amin(masked_CSC_TRAVEL_MATRIX,axis=1) # distance to closest solution CSC
           
            node_choice[:,2]=np.argmin(masked_TRAVEL_MATRIX,axis=1)# ID for the nearest solution hospital
            node_choice[:,3]=csc_id_from_solution_hospitals[0,np.argmin(masked_CSC_TRAVEL_MATRIX,axis=1)] #store the CSC ID (from the hospital solution list) that has the shortest distance from the first hosptial

#'KP CAN CALCULATE THE NEAREST CSC FROM EACH HASU ONCE AND LOOK IT UP
            node_choice[:,4]=TIME_csc_nearest_to_HASU[np.int_(node_choice[:,2])]
            node_choice[:,5]=ID_csc_nearest_to_HASU[np.int_(node_choice[:,2])]

  #          print('B',datetime.datetime.now().strftime("%H:%M:%S.%f"))

            delayed_thrombectomy=(node_choice[:,0]+node_choice[:,4])-node_choice[:,1]
            mask_go_to_CSC=delayed_thrombectomy>ALLOWABLE_THROMBECTOMY_DELAY
            node_results[:,0]=np.invert(mask_go_to_CSC)*node_choice[:,0]+mask_go_to_CSC*node_choice[:,1]
            first_hospital_masked_ID=np.invert(mask_go_to_CSC)*node_choice[:,2]+mask_go_to_CSC*node_choice[:,3]

#            print('C',datetime.datetime.now().strftime("%H:%M:%S.%f"))

        elif ALLOWABLE_THROMBOLYSIS_DELAY>0:
#MODEL3: CSC has larger catchment

#Take the min distance for the CSC, and compare with the min distance of the other hosptials whilst taking the allowance into consideration
                                                                           
            node_choice=np.zeros((len(NODE_ADMISSIONS),10))

            #FIND CLOSEST HOSPITAL FROM ALL IN SOLUTION
            node_choice[:,0]=np.amin(masked_TRAVEL_MATRIX,axis=1) # distance to closest solution hospital
            node_choice[:,1]=np.amin(masked_CSC_TRAVEL_MATRIX,axis=1) # distance to closest solution CSC
            
            node_choice[:,2]=np.argmin(masked_TRAVEL_MATRIX,axis=1)# ID for the nearest solution hospital
            node_choice[:,3]=csc_id_from_solution_hospitals[0,np.argmin(masked_CSC_TRAVEL_MATRIX,axis=1)] #store the CSC ID (from the hospital solution list) that has the shortest distance from the first hosptial

#Binary mask for each patient if they go to a CSC first, or a non CSC first
            mask_go_to_CSC=(node_choice[:,1]-ALLOWABLE_THROMBOLYSIS_DELAY)<node_choice[:,0]

#Use Binary mask to take the distance travel to first destination
            node_results[:,0]=np.invert(mask_go_to_CSC)*node_choice[:,0]+mask_go_to_CSC*node_choice[:,1]
#Use Binary mask to take the ID to first destination
            first_hospital_masked_ID=np.invert(mask_go_to_CSC)*node_choice[:,2]+mask_go_to_CSC*node_choice[:,3]
                                                                            
                                                                            
        elif ALLOWABLE_THROMBOLYSIS_DELAY<=0:
#MODEL 1 (nearest) OR MODEL2 (go to CSC if within set time).  Model 2 needs information from Model 1 first

#MODEL 1
        #Distance to closest hospital [MODDEL1].  

 #           print('MODEL 1&2 START',datetime.datetime.now().strftime("%H:%M:%S.%f"))

            first_hospital_masked_ID=np.argmin(masked_TRAVEL_MATRIX,axis=1) # ID to first hospital
                
  #          print('D12',datetime.datetime.now().strftime("%H:%M:%S.%f"))

            node_results[:,0]=np.amin(masked_TRAVEL_MATRIX,axis=1) # distance to first hospital (includes any adjustments for the CSC in model 3 - to be rectified in next step)
 
            if ALLOWABLE_THROMBOLYSIS_DELAY<0:
#MODEL 2 (go to CSC if within set time)
#            if ALLOWABLE_THROMBOLYSIS_DELAY<0:#GO TO CSC IF WITHIN x MINUTES, REGARDLESS OF ANOTHER CLOSER CENTRE.  ELSE CHOOSE CLOSEST

 #               print('MODEL 2',datetime.datetime.now().strftime("%H:%M:%S.%f"))

                csc_nearest_distance=np.amin(masked_CSC_TRAVEL_MATRIX,axis=1) # distance to closest solution CSC
                csc_nearest_ID=csc_id_from_solution_hospitals[0,np.argmin(masked_CSC_TRAVEL_MATRIX,axis=1)] #store the CSC ID (from the hospital solution list) that has the shortest distance from the first hosptial

#             "mask_go_to_CSC" in this part identifies patients that go to CSC because < n minutes, and not necessarily due to being closest.  The next use of this array stores whether go to CSC or not, as more patients may go to a CSC as it's the closest even though it's > n minutes
                mask_go_to_CSC=csc_nearest_distance<=-ALLOWABLE_THROMBOLYSIS_DELAY
#           Take the distance from the CSC if go there as < fixed distance... else take it from the min of the full hospital options
                node_results[:,0]=np.invert(mask_go_to_CSC)*node_results[:,0]+mask_go_to_CSC*csc_nearest_distance

#                print('F',datetime.datetime.now().strftime("%H:%M:%S.%f"))

#Overwrite the "first_hospital_masked_ID" for the CSC ID when that is <45mins
                first_hospital_masked_ID=np.invert(mask_go_to_CSC)*first_hospital_masked_ID+mask_go_to_CSC* csc_nearest_ID#node_choice[:,3]


#FOR ALL MODELS, NOW HAVE THE FIRST LOCATION THE PATIENTS ATTEND, SO NOW ALL GETTING THROMBOLYSIS. CHECK HOW MANY SATISFY THE TARGET DISTANCES:

#        print('H',datetime.datetime.now().strftime("%H:%M:%S.%f"))
 
        node_results[:,1]=node_results[:,0]<=THROMBOLYSIS_TARGET_DISTANCE_1 # =1 if target distance 1 met
        node_results[:,2]=node_results[:,0]<=THROMBOLYSIS_TARGET_DISTANCE_2 # =1 if target distance 2 met
        node_results[:,3]=node_results[:,0]<=THROMBOLYSIS_TARGET_DISTANCE_3 # =1 if target distance 3 met

 #       print('I',datetime.datetime.now().strftime("%H:%M:%S.%f"))
    
#THROMBECTOMY   #THROMBECTOMY   #THROMBECTOMY   #THROMBECTOMY

#THIS SECTION DETERMINES WHERE TO GO FOR THROMBECTOMY.  EITHER ALREADY AT A CSC, OR AT A HASU AND NEED A TRANSFER TO NEAREST CSC FROM HASU
        first_hospital_masked_ID=first_hospital_masked_ID.astype(int)
#https://stackoverflow.com/questions/19676538/numpy-array-assignment-with-copy
#https://stackoverflow.com/questions/6431973/how-to-copy-data-from-a-numpy-array-to-another?rq=1

#Initialise arrays
        csc_attended_ID = np.empty_like (first_hospital_masked_ID)
        np.copyto(csc_attended_ID,first_hospital_masked_ID)
        transfer_travel=np.empty(0)
        thrombolysis_to_thrombectomy_time=np.empty(0)
        csc_attended_ID = np.empty_like (first_hospital_masked_ID)
        thrombectomy_admissions_transferred = np.empty_like (NODE_ADMISSIONS)#Initialise

#Use NUMPY FANCY INDEXING to populate the CSC and TIME using the first_hospital_masked_ID array (which holds the indices want for each patient)
        transfer_travel=TIME_csc_nearest_to_HASU[first_hospital_masked_ID]
        csc_attended_ID= ID_csc_nearest_to_HASU[first_hospital_masked_ID]

#Which patients are needing a transfer for thrombectomy?  i.e which have a different 1st and 2nd hospital (who is not at a CSC for hteir first hosptial)
        mask_transfer_patient=csc_attended_ID!=first_hospital_masked_ID#Boolean storing whether each patient is moving (=true, has a different first and second hospital location) or not moving (= false, has same first ans second hospital)
#For the patients that are needing a transfer, store the number of admissions (usually 10% of stroke admissions)
        thrombectomy_admissions_transferred=mask_transfer_patient*(NODE_ADMISSIONS*proportion_eligible_thrombectomy)#Populate with admissions if that patient location has a different first and second hospital location
#Record thrombectomy admissions transfering to CSC
        transferred_admissions=np.bincount(np.int_(csc_attended_ID),weights=thrombectomy_admissions_transferred) # np.bincount with weights sums
#Record thrombectomy admissions leaving HASU
        transferred_admissions=transferred_admissions-np.bincount(np.int_(first_hospital_masked_ID),weights=thrombectomy_admissions_transferred) # np.bincount with weights sums


 #       print('N',datetime.datetime.now().strftime("%H:%M:%S.%f"))

        thrombolysis_to_thrombectomy_time=transfer_travel+INHOSPITAL_TRANSFER+DOOR_TO_GROINdir
#        print('O',datetime.datetime.now().strftime("%H:%M:%S.%f"))

        node_results[:,4]=first_hospital_masked_ID # stores hospital ID in case table needs to be exported later, but bincount below doesn't work when stored in NumPy array (which defaults to floating decimal)
    
        # Create matrix of number of admissions to each hospital (first hospital)
        thrombolysis_admissions=np.bincount(np.int_(first_hospital_masked_ID),weights=NODE_ADMISSIONS) # np.bincount with weights sums
        thrombolysis_admissions_matrix[i,mask_hospitals_in_solution]=thrombolysis_admissions#putting the hospital admissions into a matrix with column per hospital, row per solution.  Used to output to sheet

  #      print('Q',datetime.datetime.now().strftime("%H:%M:%S.%f"))

        # Create matrix of number of admissions to each hospital (CSC)
        #csc_attended_ID use the same ID allocated to each hosptial as for the solution hospitals, and so not start from 0 for the first (i/e could just be ID 5 as a CSC) so use the "mask" to put the values to the matrix, and not "mask_full_CSC"
        thrombectomy_admissions[0:np.int_((np.amax(csc_attended_ID)+1))]=np.bincount(np.int_(csc_attended_ID),weights=NODE_ADMISSIONS)*proportion_eligible_thrombectomy # np.bincount with weights sums

   #     print('R',datetime.datetime.now().strftime("%H:%M:%S.%f"))
        thrombectomy_admissions_matrix[i,mask_hospitals_in_solution]=thrombectomy_admissions#putting the hospital admissions into a matrix with column per hospital, row per solution.  Used to output to sheet

    #    print('S',datetime.datetime.now().strftime("%H:%M:%S.%f"))

        transferred_admissions_matrix[i,mask_hospitals_in_solution]=transferred_admissions#putting the hospital admissions into a matrix with column per hospital, row per solution.  Used to output to sheet

   #     print('T',datetime.datetime.now().strftime("%H:%M:%S.%f"))

        # record closest hospital (unused)
        node_results[:,5]=np.take(thrombolysis_admissions,np.int_(first_hospital_masked_ID)) # Lookup admissions to the thrombectomy hospital patient attends
        node_results[:,12]=np.take(thrombectomy_admissions,np.int_(csc_attended_ID))  # Lookup admissions to the thrombectomy hospital patient attends
        node_results[:,6]=node_results[:,5]>TARGET_THROMBOLYSIS_ADMISSIONS # =1 if admissions target met
                    
        node_results[:,13]=node_results[:,0]+transfer_travel
                    
        # Calculate average distance by multiplying node distance * admission numbers and divide by total admissions
        if 1 in pareto_include or CALC_ALL:
            weighted_distances=np.multiply(node_results[:,0],NODE_ADMISSIONS)
            average_distance=np.sum(weighted_distances)/TOTAL_ADMISSIONS
            score_matrix[i,1]=average_distance
        
        # Max distance for any one patient
        if 2 in pareto_include or CALC_ALL:
            score_matrix[i,2]=np.max(node_results[:,0])
        
        # Max, min and max/min number of thrombolysis admissions to each hospital
        if 3 in pareto_include or CALC_ALL:
            score_matrix[i,3]=np.max(thrombolysis_admissions)
        if 4 in pareto_include or CALC_ALL:
            score_matrix[i,4]=np.min(thrombolysis_admissions)
        if 5 in pareto_include or CALC_ALL:
            if score_matrix[i,4]>0:
                score_matrix[i,5]=score_matrix[i,3]/score_matrix[i,4]
            else:
                score_matrix[i,5]=0

        # Calculate proportion patients within target distance/time
        if 6 in pareto_include or CALC_ALL:
            score_matrix[i,6]=np.sum(NODE_ADMISSIONS[node_results[:,0]<=THROMBOLYSIS_TARGET_DISTANCE_1])/TOTAL_ADMISSIONS
        if 7 in pareto_include or CALC_ALL:
            score_matrix[i,7]=np.sum(NODE_ADMISSIONS[node_results[:,0]<=THROMBOLYSIS_TARGET_DISTANCE_2])/TOTAL_ADMISSIONS
        if 8 in pareto_include or CALC_ALL:
            score_matrix[i,8]=np.sum(NODE_ADMISSIONS[node_results[:,0]<=THROMBOLYSIS_TARGET_DISTANCE_3])/TOTAL_ADMISSIONS

        # Calculate proportion patients attending hospital with target admissions
        if 9 in pareto_include or CALC_ALL:
            score_matrix[i,9]=np.sum(thrombolysis_admissions[thrombolysis_admissions>=TARGET_THROMBOLYSIS_ADMISSIONS])/TOTAL_ADMISSIONS
        if 10 in pareto_include or CALC_ALL:
            # Sum patients who meet distance taregts
            node_results[:,7]=(node_results[:,1]+node_results[:,6])==2 # true if admissions and target distance 1 both met
            sum_patients_addmissions_distance1_met=np.sum(NODE_ADMISSIONS[node_results[:,7]==1])
            score_matrix[i,10]=sum_patients_addmissions_distance1_met/TOTAL_ADMISSIONS
        if 11 in pareto_include or CALC_ALL:
            # Sum patients who meet distance taregts
            node_results[:,8]=(node_results[:,2]+node_results[:,6])==2 # true if admissions and target distance 2 both met
            sum_patients_addmissions_distance2_met=np.sum(NODE_ADMISSIONS[node_results[:,8]==1])
            score_matrix[i,11]=sum_patients_addmissions_distance2_met/TOTAL_ADMISSIONS
        if 12 in pareto_include or CALC_ALL:
            # Sum patients who meet distance taregts
            node_results[:,9]=(node_results[:,3]+node_results[:,6])==2 # true if admissions and target distance 3 both met
            sum_patients_addmissions_distance3_met=np.sum(NODE_ADMISSIONS[node_results[:,9]==1])
            score_matrix[i,12]=sum_patients_addmissions_distance3_met/TOTAL_ADMISSIONS
        if 13 in pareto_include or CALC_ALL:
            # Clinical benefit
            score_matrix[i,13]=0
        if 14 in pareto_include or CALC_ALL:
            #Time to thrombolysis
   #         print('U',datetime.datetime.now().strftime("%H:%M:%S.%f"))
            node_results[:,10]=node_results[:,0]+DOOR_TO_NEEDLE
            score_matrix[i,14]=np.sum(NODE_ADMISSIONS[node_results[:,10]<=TARGET_THROMBOLYSIS_TIME])/TOTAL_ADMISSIONS
        if 15 in pareto_include or CALC_ALL:
            #Time to thrombectomy = Journey to location1 + thrombolysis at location1 + transfer_time(if going to different location) + door_to_groin (different if in CSC for throbolysis, or transferred in)
  #          print('V',datetime.datetime.now().strftime("%H:%M:%S.%f"))
            node_results[:,11]=node_results[:,10] + thrombolysis_to_thrombectomy_time 
            score_matrix[i,15]=np.sum(NODE_ADMISSIONS[node_results[:,11]<=TARGET_THROMBECTOMY_TIME])/TOTAL_ADMISSIONS            

        # Calculate proportion patients travel within target distance/time to get to their CSC location (for thrombectomy)
        if 16 in pareto_include or CALC_ALL:
  #          print('W',datetime.datetime.now().strftime("%H:%M:%S.%f"))
            score_matrix[i,16]=np.sum(NODE_ADMISSIONS[node_results[:,13]<=THROMBECTOMY_TARGET_DISTANCE_1])/TOTAL_ADMISSIONS
        if 17 in pareto_include or CALC_ALL:
#            print('X',datetime.datetime.now().strftime("%H:%M:%S.%f"))
           score_matrix[i,17]=np.sum(NODE_ADMISSIONS[node_results[:,13]<=THROMBECTOMY_TARGET_DISTANCE_2])/TOTAL_ADMISSIONS
        if 18 in pareto_include or CALC_ALL:
   #         print('Y',datetime.datetime.now().strftime("%H:%M:%S.%f"))
            score_matrix[i,18]=np.sum(NODE_ADMISSIONS[node_results[:,13]<=THROMBECTOMY_TARGET_DISTANCE_3])/TOTAL_ADMISSIONS
        
        #Proportion of CSC that have > TARGET THROMBECTOMY ADMISSIONS
        if 19 in pareto_include or CALC_ALL:
 #           print('Z',datetime.datetime.now().strftime("%H:%M:%S.%f"))
            score_matrix[i,19]=np.sum(thrombectomy_admissions[:]>=TARGET_THROMBECTOMY_ADMISSIONS)/CSC_COUNT

        # Max and min of thrombectomy admissions to each hospital
        if 20 in pareto_include or CALC_ALL:
            score_matrix[i,20]=np.max(thrombectomy_admissions[mask_csc_from_solution_hospitals])
        if 21 in pareto_include or CALC_ALL:
            score_matrix[i,21]=np.min(thrombectomy_admissions[mask_csc_from_solution_hospitals])
	
# average time to thrombectomy (including door-in-door-out delay)
       	if 22 in pareto_include or CALC_ALL:
            weighted_distances=np.multiply(node_results[:,13],NODE_ADMISSIONS)
            average_distance=np.sum(weighted_distances)/TOTAL_ADMISSIONS
            score_matrix[i,22]=average_distance 


#output this table for each hospital:
#Hospital/Direct admissions (thrombolysis)/Transferred in admissions (for thrombectomy)/Thrombectomy admissions/Total patients

    #Calculate clinical benefit: Emberson and Lee
    #Use 115 mins for the onset til travelling in ambulance (30 mins onset to call + 40 mins call to travel + 45 mins door to needle) + ? travel time (as deterined by the combination of hospital open)
#        if 13 in pareto_include or CALC_ALL:
#            onset_to_treatment_time = distancekm_to_timemin(node_results[:,0])+115
#            #constant to be used in the equation
#            factor=(0.2948/(1 - 0.2948))
#            #Calculate the adjusted odds ratio
#            clinical_benefit=np.array(factor*np.power(10, (0.326956 + (-0.00086211 * onset_to_treatment_time))))
#            # Patients that exceed the licensed onset to treatment time, set to a zero clinical benefit
#            clinical_benefit[onset_to_treatment_time>270]=0 
#            #Probabilty of good outcome per node
#            clinical_benefit = (clinical_benefit / (1 + clinical_benefit)) - 0.2948
#            #Number of patients with a good outcome per node
#            clinical_benefit = clinical_benefit*NODE_ADMISSIONS
#            score_matrix[i,13]=np.sum(clinical_benefit)/TOTAL_ADMISSIONS *100

    #hospital_admissions_matrix[i,:]=np.transpose(hospital_admissions)#putting the column into a row in the matrix
    #np.savetxt('output/admissions_test.csv',hospital_admissions_matrix[i,:],delimiter=',',newline='\n')
    
    
    
    total_patients_matrix=thrombolysis_admissions_matrix+transferred_admissions_matrix#putting the hospital admissions into a matrix with column per hospital, row per solution.  Used to output to sheet
    return (score_matrix,thrombolysis_admissions_matrix,transferred_admissions_matrix,thrombectomy_admissions_matrix,total_patients_matrix)
    
def unique_rows(a): # stolen off the interwebs
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))
    
def fix_hospital_status(l_population,l_HOSPITAL_STATUS):
    #Takes the 5th column from the hospital.csv file and if "1" then open, "-1" then closed
    HOSPITAL_STATUS_POPULATION=np.repeat(l_HOSPITAL_STATUS,len(l_population[:,0]),axis=0)#repeat the row "len(child_population[:,0])" number of times, so have 1 per solution row (matching the size of the child_population matrix)
    l_population[HOSPITAL_STATUS_POPULATION==1]=1 # Fixes the open hospitals to have a value 1
    l_population[HOSPITAL_STATUS_POPULATION==2]=1 # Fixes the open hospitals to have a value 1
    l_population[HOSPITAL_STATUS_POPULATION==-1]=0 # Fixes the closed hospitals to have a value 0
    return l_population
    
    
def f_location_crossover(l_parent, l_MAXCROSSOVERPOINTS,l_CHROMOSOMELENGTH):
   
    number_crossover_points=rn.randint(1,l_MAXCROSSOVERPOINTS) # random, up to max
    crossover_points=rn.sample(range(1,l_CHROMOSOMELENGTH), number_crossover_points) # pick random crossover points in gene, avoid first position (zero position)
    crossover_points=np.append([0],np.sort(crossover_points)) # zero appended at front for calucation of interval to first crossover
    intervals=crossover_points[1:]-crossover_points[:-1] # this gives array of number of ones, zeros etc in each section.
    intervals=np.append([intervals],[l_CHROMOSOMELENGTH-np.amax(crossover_points)]) # adds in last interval of last cross-over to end of gene
    
    # Build boolean arrays for cross-overs
    current_bool=True # sub sections will be made up of repeats of boolean true or false, start with true

    # empty list required for append
    selection1=[] 

    for interval in intervals: # interval is the interval between crossoevrs (stored in 'intervals')
        new_section=np.repeat(current_bool,interval) # create subsection of true or false
        current_bool=not current_bool # swap true to false and vice versa
        selection1=np.append([selection1],[new_section]) # add the new section to the existing array
    
    selection1=np.array([selection1],dtype=bool) # **** not sure why this is needed but selection1 seems to have lost boolean nature
    selection2=np.invert(selection1) #  invert boolean selection for second cross-over product

    crossover1=np.choose(selection1,l_parent) # choose from parents based on selection vector
    crossover2=np.choose(selection2,l_parent)

    children=np.append(crossover1,crossover2,axis=0)
    
    return(children)
  
def f_uniform_crossover(l_parent, l_CHROMOSOMELENGTH):
#UNIFORM CROSSOVER MEANS EACH GENE HAS EQUAL CHANCE TO COME FROM EACH PARENT
    fromparent1=np.random.random_integers(0,1,(1,l_CHROMOSOMELENGTH)) # create array of 1 rows and chromosome columns and fill with 0 or 1 for which parent to take the gene from
    fromparent1=np.array(fromparent1,dtype=bool)
    fromparent2=np.invert(fromparent1)#opposite of fromparent1
    crossover1=np.choose(fromparent1,l_parent) # choose from the 2 parents based on select vector
    crossover2=np.choose(fromparent2,l_parent)
    children=np.append(crossover1,crossover2,axis=0)
    return(children)

def distancekm_to_timemin(distancekm):
#    Using EV's 5th order polynomial
#    speed=np.array(25.9364 + 0.740692*distance -0.00537274*np.power(distance,2)+ 0.000019121*np.power(distance,3)-0.0000000319161*np.power(distance,4)+0.0000000000199508*np.power(distance,5))
#    Using EV's 8th order polynomial, max distance is 400 (set all speeds for distances over 400, to if 400), otherwise the equation behaves in an odd way
    distancekm[distancekm>400]=400
    speedkmhr=np.array(17.5851+1.41592*distancekm-0.0212389*(distancekm**2)+0.000186836*(distancekm**3)-0.000000974605*(distancekm**4)+0.0000000030356*(distancekm**5)-0.00000000000551583*(distancekm**6)+0.00000000000000537742*(distancekm**7)-0.00000000000000000216925*(distancekm**8))
    timemin=np.array((distancekm/speedkmhr)*60)

#    time=np.array(25.9364 + 0.740692*distance -0.00537274*math.pow(distance,2)+ 1.9121e-05*math.pow(distance,3) -3.19161e-08*math.pow(distance,4)+1.99508e-11*math.pow(distance,5))
#    Using EV's 1st order polynomial
#    time=np.array(52.0884 + 0.0650975*distance)
    return(timemin)   

def calculate_epsilon(l_normscores):
#did use three inputs: (l_nscores,l_normscores,l_population):

#l_WHICH_EPSILON: Set to 0 for just max, 1 for just min, 2 for min and max, 3 for none

# The reference point for the Epsilon calculation "Utopia" in this instance is 1 for all dimensions.
#    utopia=np.ones(l_nscores*POPULATION).reshape((POPULATION,l_nscores))

#The algotihm requires all the scores to be divided by the utopia, but while it's 1 then no need.  Rows of code included below for completelness for when utopia <>1, but at the mo not required.
#    npdivide = l_normscores/utopia
#    epsilon = np.amin(npdivide,1)#the min score for each of the population.  Get each person to put forward their worst attribute
    epsilon = np.amin(l_normscores,1)#the min score for each of the population.  Get each person to put forward their worst attribute
    return (np.amin(epsilon),np.amax(epsilon))

def is_epsilon_static(l_epsilonmin, l_epsilonmax,l_generation,l_CONSTANT_EPSILON,l_WHICH_EPSILON):
    if l_WHICH_EPSILON==0:
        #need just epsilon max not change over CONSTANT_EPSILON generations
        return check_epsilon_static(l_epsilonmax,l_generation,l_CONSTANT_EPSILON)
    elif l_WHICH_EPSILON==1:
        #need just epsilon min not change over CONSTANT_EPSILON generations
        return check_epsilon_static(l_epsilonmin,l_generation,l_CONSTANT_EPSILON)
    elif l_WHICH_EPSILON==2:
        #need both epsilon min and epsilon max to not change
        if check_epsilon_static(l_epsilonmax,l_generation,l_CONSTANT_EPSILON):
            #only need to check epsilonmin if epsilonmax is satisfied
            return check_epsilon_static(l_epsilonmin,l_generation,l_CONSTANT_EPSILON)
        else:
            return False            

def check_epsilon_static(l_epsilonarray,l_generation,l_CONSTANT_EPSILON):
    i=0
    while l_epsilonarray[l_generation-i]==l_epsilonarray[l_generation-i-1] and i<l_CONSTANT_EPSILON:
       i+=1 
   
    if i==l_CONSTANT_EPSILON:
        #reached static population for 3 generations.  KP check: Can I set generation to GENERATIONS?
        #else do a while on a boolean
        return True #jumps over code for remaining generation
    else:
        return False #jumps over code for remaining generation

def csc_travel_allowance(l_LENGTH,l_HOSPITAL_STATUS,l_ALLOWANCE):
    #Needed for model 3: Patients go to closest unit, but adjust travel time so n mins in favour of CSC
    #Takes the 9th column from the hospital.csv file and if "2" then a CSC and so reduce the travel time by the user defined ALLOWANCE (so patients go there if it's within being that much further than the nearest HASU)
    CSC_TRAVEL_ALLOWANCE=np.repeat(l_HOSPITAL_STATUS,l_LENGTH,axis=0)#repeat the row "len(child_population[:,0])" number of times, so have 1 per solution row (matching the size of the child_population matrix)
    if l_ALLOWANCE>0:#if minus (say, -45) then this states that patients will go to CSC regardless of if a HASU is closer if the travel is <45 mins
        CSC_TRAVEL_ALLOWANCE[CSC_TRAVEL_ALLOWANCE==2]=l_ALLOWANCE
    else:
        CSC_TRAVEL_ALLOWANCE[CSC_TRAVEL_ALLOWANCE==2]=0
    CSC_TRAVEL_ALLOWANCE[CSC_TRAVEL_ALLOWANCE<2]=0
    return CSC_TRAVEL_ALLOWANCE

def csc_travel_allowance1(l_LENGTH,l_HOSPITAL_STATUS,l_ALLOWANCE):
    #ONLY CALLED WHEN L_ALLOWANCE>0
    #Needed for model 3: Patients go to closest unit, but adjust travel time so n mins in favour of CSC
    #Takes the 9th column from the hospital.csv file and if "2" then a CSC and so reduce the travel time by the user defined ALLOWANCE (so patients go there if it's within being that much further than the nearest HASU)
    CSC_TRAVEL_ALLOWANCE=np.repeat(l_HOSPITAL_STATUS,l_LENGTH,axis=0)#repeat the row "len(child_population[:,0])" number of times, so have 1 per solution row (matching the size of the child_population matrix)
    CSC_TRAVEL_ALLOWANCE[CSC_TRAVEL_ALLOWANCE<2]=0
    CSC_TRAVEL_ALLOWANCE[CSC_TRAVEL_ALLOWANCE==2]=l_ALLOWANCE
    return CSC_TRAVEL_ALLOWANCE

def travel_matrix_adjust_csc(l_TRAVEL_MATRIX,l_HOSPITAL_STATUS,l_ALLOWANCE):
    #Takes the 9th column from the hospital.csv file and if "2" then a CSC and so reduce the travel time by the user defined ALLOWANCE (so patients go there if it's within being that much further than the nearest HASU)
    if l_ALLOWANCE>0:#if minus (say, -45) then this states that patients will go to CSC regardless of if a HASU is closer if the travel is <45 mins
        HOSPITAL_STATUS_POPULATION=np.repeat(l_HOSPITAL_STATUS,len(l_TRAVEL_MATRIX[:,0]),axis=0)#repeat the row "len(child_population[:,0])" number of times, so have 1 per solution row (matching the size of the child_population matrix)
        HOSPITAL_STATUS_POPULATION[HOSPITAL_STATUS_POPULATION==2]=l_ALLOWANCE
        HOSPITAL_STATUS_POPULATION[HOSPITAL_STATUS_POPULATION<2]=0
        l_TRAVEL_MATRIX=l_TRAVEL_MATRIX-HOSPITAL_STATUS_POPULATION#l_TRAVEL_MATRIX-l_ALLOWANCE # For any CSC, adjust the travel so that a patient will go there over a nearer HASU by this allownace (say 15 minutes)
    return l_TRAVEL_MATRIX

def hasu_add_travel(l_LENGTH,l_HOSPITAL_STATUS,l_ADD):
    #Needed for model 3: Patients go to closest unit, but adjust travel time so n mins in favour of CSC
    #Takes the 9th column from the hospital.csv file and if "2" then a CSC and so reduce the travel time by the user defined ALLOWANCE (so patients go there if it's within being that much further than the nearest HASU)
    HASU_ADD_TRAVEL=np.repeat(l_HOSPITAL_STATUS,l_LENGTH,axis=0)#repeat the row "len(child_population[:,0])" number of times, so have 1 per solution row (matching the size of the child_population matrix)
    HASU_ADD_TRAVEL[HASU_ADD_TRAVEL<2]=l_ADD
    HASU_ADD_TRAVEL[HASU_ADD_TRAVEL==2]=0
    return HASU_ADD_TRAVEL

def travel_return_ID(to_length,solution_matrix,find_array):
    solution_ID=np.empty(0, dtype=int)
    for n in range(to_length):
        a = solution_matrix[n,:]#take the list of travel times for a patient to the solution hospitals
        #j = a.tolist().index(find_array[n]) # find the location of the match for the distance for the hosptial attending from the list
        solution_ID=np.append(solution_ID,a.tolist().index(find_array[n]))#store the location match, this is the hosptial ID that the patient is attending
    return solution_ID

def transfer_return_ID(to_length,solution_matrix,from_array,find_array):
    solution_ID=np.empty(0, dtype=int)
    for n in range(to_length):
        a = solution_matrix[np.int(from_array[n]),:]#take the list of travel times from the nearest solution hospital to the CSC
       # j = a.tolist().index(find_array[n]) # find the location of the match for the distance for the hosptial attending from the list
        solution_ID=np.append(solution_ID,a.tolist().index(find_array[n]))#store the location match, this is the CSC ID that the patient is attending
    return solution_ID


