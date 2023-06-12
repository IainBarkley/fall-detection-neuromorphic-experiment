import pandas as pd
import sys,os

sensor_data_dir = '../fall_detection_data/sensor_data'
out_dir = '../fall_detection_data/merged_processed'

subjects = os.listdir(sensor_data_dir)

for subject in subjects:
    print('loading labelled data for subject {}'.format(subject))

    # load the labelled data
    labelled_data_dir = '../fall_detection_data/label_data'
    subject_filename = '{}_label.xlsx'.format(subject)
    lbdf = pd.read_excel(os.path.join(os.path.dirname(__file__),labelled_data_dir,subject_filename))
    lbdf['Task Code (Task ID)'] = lbdf['Task Code (Task ID)'].ffill()
    lbdf['Description'] = lbdf['Description'].ffill()
   
    lbdf['Subject Code'] = subject
    lbdf['Task Code'] = 'T' + lbdf['Task Code (Task ID)'].str.split(' ',1,expand=True)[1].str.strip('()')
    lbdf['Trial Code'] = 'R0' + lbdf['Trial ID'].astype(str)
    lbdf['Sensor Data Code'] = lbdf['Subject Code'] + lbdf['Task Code'] + lbdf['Trial Code']

    # load the sensor data
    trial_files = [f for f in os.listdir(os.path.join(sensor_data_dir,subject)) if '.csv' in f]
    
    for i,trial_file in enumerate(trial_files):
        temp_df = pd.read_csv(os.path.join(sensor_data_dir,subject,trial_file))
        
        # create a new dataframe for the individual if this is the first sensor data file we're grabbing
        if i == 0:
            ivdf = pd.DataFrame(columns = temp_df.columns)
            
        # grab the corresponding label data
        trial_info = lbdf[ lbdf['Sensor Data Code'] == trial_file[:-4] ]
        
        if trial_info.shape[0] > 0:
            
            #fall_mask = (temp_df['Frame'] > trial_info['Fall_onset_frame']) & (temp_df['Frame'] < trial_info['Fall_impact_frame'])
            temp_df['Fall/No Fall'] = np.where( (temp_df['Frame'] > trial_info['Fall_onset_frame']) & (temp_df['Frame'] < trial_info['Fall_impact_frame']), 1, 0)
            temp_df['Fall Type'] = np.where( (temp_df['Frame'] > trial_info['Fall_onset_frame']) & (temp_df['Frame'] < trial_info['Fall_impact_frame']), trial_info['Task Code'], 0)
        else:
            temp_df['Fall/No Fall'] = 0
            temp_df['Fall Type'] = 0
        
        print(temp_df[temp_df['Fall/No Fall'] == 1].sample(10))
        ivdf = pd.concat([ivdf,temp_df],axis=0,ignore_index=True)
    
    out_filename = '{}-merged.csv'.format(subject)
    ivdf.to_csv(os.path.join(out_dir,out_filename))
