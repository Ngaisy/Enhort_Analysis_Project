import pandas as pd 
import numpy as np
import datetime
import math
import matplotlib.pyplot as plt
import itertools
class enhort_analysis():
    def __init__(self):
        self.as_tf_data = pd.read_csv('auto_stash_transfers.csv')
        self.user_data = pd.read_csv('users.csv')
        self.user_data.date_of_birth = pd.to_datetime(self.user_data.date_of_birth)
        self.user_data.created_at = pd.to_datetime(self.user_data.created_at)
        self.as_tf_data.created_at = pd.to_datetime(self.as_tf_data.created_at)
        self.date_gap = []
        self.age_gap = []
        self.state_gap = []
        self.salary_range = []
        self.user_id_by_parameter = []
        self.state_aver_income = pd.read_csv('State_aver_income.csv')
        
        
    def define_date_gap(self):
        self.user_data.created_at = pd.to_datetime(self.user_data.created_at)
        self.as_tf_data.created_at = pd.to_datetime(self.as_tf_data.created_at)
        # Find the range of the date
        l_recent_date = self.user_data.created_at.min()
        m_recent_date = self.user_data.created_at.max()

        print("\033[1m The least recent recorded date: '\033[0m'", l_recent_date)
        print("\033[1m The most recent recorded date: '\033[0m'", m_recent_date)
        
        if l_recent_date.dayofweek != 1:
            days_to_nx_mday = 7 - l_recent_date.dayofweek
            self.date_gap.append(l_recent_date.normalize())
            l_recent_date = l_recent_date + datetime.timedelta(days=days_to_nx_mday)

        for i in range(0,(m_recent_date - l_recent_date).days + 7,7):
            self.date_gap.append((l_recent_date + datetime.timedelta(days=i)).normalize())
        print("\033[1m" + "\n The weekly gap:"+'\033[0m')
        print(self.date_gap)

    def percentage_allusers_mk_ast(self):
        # Step 1 Make sure all the id in as_tf_data is in user_data
        test = self.as_tf_data.user_id.unique()
        test2 = self.user_data.id.unique()
        for i in test:
            if i not in test2:
                print(False)
                break

        # Release memory
        del test
        del test2

        # Step 2 Calculate the percentage of users make auto-stash transfer
        pert_mk_ato_stsh = self.as_tf_data.user_id.nunique()/self.user_data.id.nunique()

        # Step 3 Print the percentage of users make auto-stash transfer
        return pert_mk_ato_stsh

        
    def gp_user_by_created_wk(self):
        as_tf_data_id = self.as_tf_data['user_id'].unique()
        ur_id_id_cr = self.user_data[['id','created_at']]
        self.user_id_by_parameter = list(ur_id_id_cr.groupby(pd.cut(ur_id_id_cr.created_at,self.date_gap)).id)
    
    def each_ast_user_age(self):
        self.user_data.date_of_birth = pd.to_datetime(self.user_data.date_of_birth)
        user_id_astcrat_astdob = pd.merge(self.as_tf_data[['user_id','created_at']],self.user_data[['user_id','date_of_birth']],how='left',on = 'user_id')
        user_id_astcrat_astdob['age'] = user_id_astcrat_astdob.created_at.apply(lambda x:x.year) - user_id_astcrat_astdob.date_of_birth.apply(lambda x:x.year) 

        add_one_year_m = user_id_astcrat_astdob.created_at.apply(lambda x:x.month) == user_id_astcrat_astdob.date_of_birth.apply(lambda x:x.month)
        add_one_year_d = user_id_astcrat_astdob.created_at.apply(lambda x:x.day) < user_id_astcrat_astdob.date_of_birth.apply(lambda x:x.day)
        user_id_astcrat_astdob['age'] -= add_one_year_d * add_one_year_m

        add_one_year_m = user_id_astcrat_astdob.created_at.apply(lambda x:x.month) < user_id_astcrat_astdob.date_of_birth.apply(lambda x:x.month)
        user_id_astcrat_astdob['age'] -= add_one_year_m

        min_age = user_id_astcrat_astdob.age.min()
        max_age = user_id_astcrat_astdob.age.max()
        self.age_gap = np.arange(min_age-min_age%5,max_age+5-max_age%5+10,10)
        self.user_id_by_parameter = list(user_id_astcrat_astdob.groupby(pd.cut(user_id_astcrat_astdob.age,self.age_gap)))
    
    def gp_user_by_age(self):
        self.user_data.date_of_birth = pd.to_datetime(self.user_data.date_of_birth)
        self.user_data.created_at = pd.to_datetime(self.user_data.created_at)
        self.user_data['age'] = self.user_data.created_at.apply(lambda x:x.year) - self.user_data.date_of_birth.apply(lambda x:x.year)
        
        add_one_year_m = self.user_data.created_at.apply(lambda x:x.month) == self.user_data.date_of_birth.apply(lambda x:x.month)
        add_one_year_d = self.user_data.created_at.apply(lambda x:x.day) < self.user_data.date_of_birth.apply(lambda x:x.day)
        self.user_data['age'] -= add_one_year_d * add_one_year_m

        add_one_year_m = self.user_data.created_at.apply(lambda x:x.month) < self.user_data.date_of_birth.apply(lambda x:x.month)
        self.user_data['age'] -= add_one_year_m
        
        min_age = self.user_data.age.min()
        max_age = self.user_data.age.max()
        self.age_gap = np.arange(min_age-min_age%5,max_age+5-max_age%5+10,10)
        ur_id_id_age = self.user_data[['id','age']]
        self.user_id_by_parameter = list(ur_id_id_age.groupby(pd.cut(ur_id_id_age.age,self.age_gap)).id)
        
    def gp_user_by_state(self):
        # Group users by state
        max_state_inc = self.state_aver_income['Real Income'].max()
        min_state_inc = self.state_aver_income['Real Income'].min()
        self.salary_range = np.arange(math.floor(min_state_inc/10000)*10000,math.ceil(max_state_inc/10000)*10000+10000,5000)
        self.state_gap = list(self.state_aver_income.groupby(pd.cut(self.state_aver_income['Real Income'],self.salary_range)))
        user_id_state = pd.merge(self.user_data,self.state_aver_income[['state','Real Income']],how='left',on=['state'])
        self.user_id_by_parameter = list(user_id_state.groupby(pd.cut(user_id_state['Real Income'],self.salary_range)).id)
 
        
    def percentage_user_mk_ast(self): # Question 2a
        pert_by_wk = []
        as_tf_data_id = self.as_tf_data['user_id'].unique()
        for i in range(len(self.user_id_by_parameter)):
            if len(self.user_id_by_parameter[i][1]) != 0:
                pert_by_wk.append(self.user_id_by_parameter[i][1].isin(as_tf_data_id).sum() / len(self.user_id_by_parameter[i][1]))
        return pert_by_wk
    
    def make_bar_plot(self,pert_by_wk):
        plt.bar(list(range(len(pert_by_wk))),pert_by_wk)
        plt.xticks(list(range(len(pert_by_wk))),map(lambda x:x+1,list(range(len(pert_by_wk)))))
        plt.ylim([0,max(pert_by_wk)+0.1*max(pert_by_wk)])
        
        ax = plt.gca()
        if np.mean(pert_by_wk) > 100: 
            for i, txt in enumerate(pert_by_wk):
                ax.annotate(txt, (list(range(len(pert_by_wk)))[i]-0.4,pert_by_wk[i]*1.01))
        elif np.mean(pert_by_wk) < 0.1:
            for i, txt in enumerate(pert_by_wk):
                ax.annotate("{0:.3f}".format(txt), (list(range(len(pert_by_wk)))[i]-0.4,pert_by_wk[i]*1.01))

        else:
            for i, txt in enumerate(pert_by_wk):
                ax.annotate("{0:.2f}".format(txt), (list(range(len(pert_by_wk)))[i]-0.4,pert_by_wk[i]*1.01))
        return plt
    
    def make_bar_plot_gp(self,pert_by_wk,ax):
        ax.bar(list(range(len(pert_by_wk))),pert_by_wk)
        plt.xticks(list(range(len(pert_by_wk))),map(lambda x:x+1,list(range(len(pert_by_wk)))))
        plt.ylim([0,max(pert_by_wk)+0.1*max(pert_by_wk)])

#         gca = plt.gca()
        if np.mean(pert_by_wk) > 100: 
            for i, txt in enumerate(pert_by_wk):
                ax.annotate(txt, (list(range(len(pert_by_wk)))[i]-0.4,pert_by_wk[i]*1.01))
        elif np.mean(pert_by_wk) < 0.1:
            for i, txt in enumerate(pert_by_wk):
                ax.annotate("{0:.3f}".format(txt), (list(range(len(pert_by_wk)))[i]-0.4,pert_by_wk[i]*1.01))

        else:
            for i, txt in enumerate(pert_by_wk):
                ax.annotate("{0:.2f}".format(txt), (list(range(len(pert_by_wk)))[i]-0.4,pert_by_wk[i]*1.01))
        return ax
        
    def average_ast(self): # Question b
        aveg_tf_per_user = []
        for i in range(len(self.user_id_by_parameter)):
            if len(self.user_id_by_parameter[i][1]) != 0:
                num_user_make_transfers = self.user_id_by_parameter[i][1].isin(self.as_tf_data.user_id.unique()).sum()
                aveg_tf_per_user.append(len(self.as_tf_data[self.as_tf_data['user_id'].isin(self.user_id_by_parameter[i][1])])/num_user_make_transfers)
        return aveg_tf_per_user
            
    def average_freq_ast(self): # Question  c
        # Find the days each users have used auto-stash transfer
        user_id_created_days = self.user_data[['id','created_at']]
        max_date_transfer = self.as_tf_data.groupby('user_id').apply(lambda x:x.created_at.max()).to_frame().reset_index(level = 0, inplace = False)

        user_id_created_days.columns = ['user_id','created_at']
        max_date_transfer.columns = ['user_id','created_at']

        user_id_dcr_duse = pd.merge(max_date_transfer,user_id_created_days,how='left',on=['user_id'] )
        user_id_dcr_duse['use_days'] = (user_id_dcr_duse.created_at_x - user_id_dcr_duse.created_at_y).apply(lambda x:x.days)
        user_id_days_use = user_id_dcr_duse[['user_id','use_days']]
        user_id_days_use.use_days.replace(0,1,inplace=True)
        del user_id_dcr_duse          
        
        # Find the total transfer each user makes
        user_id_num_transfer = self.as_tf_data.groupby('user_id').user_id.count()
        user_id_num_transfer = user_id_num_transfer.to_frame()
        user_id_num_transfer.columns=['total_transfer']
        user_id_num_transfer.reset_index(level=0,inplace = True)
        
        user_id_aver_freq = pd.merge(user_id_num_transfer,user_id_days_use,how='left',on=['user_id'])
        user_id_aver_freq['aver_freq'] = user_id_aver_freq.total_transfer.div(user_id_days_use.use_days,axis=0)
        

        aveg_tf_per_day = []
        for i in range(len(self.user_id_by_parameter)):
            if len(self.user_id_by_parameter[i][1]) != 0:
                aveg_tf_per_day.append((user_id_aver_freq[user_id_aver_freq['user_id'].isin(self.user_id_by_parameter[i][1])]['aver_freq'].sum())/len(self.user_id_by_parameter[i][1]))
        return aveg_tf_per_day

    def average_freq_ast_per_user(self): # Question  c
        # Find the days each users have used auto-stash transfer
        user_id_created_days = self.user_data[['id','created_at']]
        max_date_transfer = self.as_tf_data.groupby('user_id').apply(lambda x:x.created_at.max()).to_frame().reset_index(level = 0, inplace = False)

        user_id_created_days.columns = ['user_id','created_at']
        max_date_transfer.columns = ['user_id','created_at']

        user_id_dcr_duse = pd.merge(max_date_transfer,user_id_created_days,how='left',on=['user_id'] )
        user_id_dcr_duse['use_days'] = (user_id_dcr_duse.created_at_x - user_id_dcr_duse.created_at_y).apply(lambda x:x.days)
        user_id_days_use = user_id_dcr_duse[['user_id','use_days']]
        user_id_days_use.use_days.replace(0,1,inplace=True)
        del user_id_dcr_duse          
        
        # Find the total transfer each user makes
        user_id_num_transfer = self.as_tf_data.groupby('user_id').user_id.count()
        user_id_num_transfer = user_id_num_transfer.to_frame()
        user_id_num_transfer.columns=['total_transfer']
        user_id_num_transfer.reset_index(level=0,inplace = True)
        
        user_id_aver_freq = pd.merge(user_id_num_transfer,user_id_days_use,how='left',on=['user_id'])
        user_id_aver_freq['aver_freq'] = user_id_aver_freq.total_transfer.div(user_id_days_use.use_days,axis=0)
        return user_id_aver_freq
        
    def average_increase_ast(self): 
        as_tf_data_ui_ca_amt = self.as_tf_data[['user_id','created_at','amount']]
        as_tf_data_ui_ca_amt = as_tf_data_ui_ca_amt.groupby(['user_id']).apply(lambda x:x.sort_values(by=['created_at']))
        as_tf_data_ui_ca_amt = as_tf_data_ui_ca_amt.reset_index(level = 1,drop=True)

        as_tf_data_ui_amt = as_tf_data_ui_ca_amt[['user_id','amount']]
        as_tf_data_ui_amt_sft = as_tf_data_ui_amt.groupby('user_id').apply(lambda x:x.shift(1))
        
        as_tf_data_ui_amt_diff = (as_tf_data_ui_amt - as_tf_data_ui_amt_sft)
        as_tf_data_ui_amt_diff.user_id = as_tf_data_ui_amt.user_id
        as_tf_data_ui_amt_diff = as_tf_data_ui_amt_diff[-as_tf_data_ui_amt_diff.amount.isnull()]
        as_tf_data_ui_amt_diff_mean = as_tf_data_ui_amt_diff.groupby(as_tf_data_ui_amt_diff.index.get_level_values(0)).mean()

        as_tf_data_ui_amt_diff_mean.columns = ['user_id','aver_ast_diff']
        ur_id_id_cr = self.user_data[['id','created_at']]
        aver_ast_diff_wk = []
        
        for i in range(len(self.user_id_by_parameter)):
            if len(self.user_id_by_parameter[i][1]) != 0:
                aver_ast_diff_wk.append(
                    as_tf_data_ui_amt_diff_mean[as_tf_data_ui_amt_diff_mean['user_id'].isin(self.user_id_by_parameter[i][1])]
                    .aver_ast_diff.mean())

        return aver_ast_diff_wk

    def average_increase_ast_per_user(self): 
        as_tf_data_ui_ca_amt = self.as_tf_data[['user_id','created_at','amount']]
        as_tf_data_ui_ca_amt = as_tf_data_ui_ca_amt.groupby(['user_id']).apply(lambda x:x.sort_values(by=['created_at']))
        as_tf_data_ui_ca_amt = as_tf_data_ui_ca_amt.reset_index(level = 1,drop=True)

        as_tf_data_ui_amt = as_tf_data_ui_ca_amt[['user_id','amount']]
        as_tf_data_ui_amt_sft = as_tf_data_ui_amt.groupby('user_id').apply(lambda x:x.shift(1))
        
        as_tf_data_ui_amt_diff = (as_tf_data_ui_amt - as_tf_data_ui_amt_sft)
        as_tf_data_ui_amt_diff.user_id = as_tf_data_ui_amt.user_id
        as_tf_data_ui_amt_diff = as_tf_data_ui_amt_diff[-as_tf_data_ui_amt_diff.amount.isnull()]
        as_tf_data_ui_amt_diff_mean = as_tf_data_ui_amt_diff.groupby(as_tf_data_ui_amt_diff.index.get_level_values(0)).mean()

        as_tf_data_ui_amt_diff_mean.columns = ['user_id','aver_ast_diff']
        return as_tf_data_ui_amt_diff_mean

    def calculate_churn_rate(self): # all the user stay in atf until 12/30/2016
        test = self.as_tf_data.groupby('user_id').apply(lambda x:x.sort_values(by='created_at'))
        test1 = test.reset_index(level = 1,drop=True)
        test2 = test1.created_at.apply(lambda x:x.normalize())
        test3 = test2.groupby(test.index.get_level_values(0)).shift(1)
        test4 = (test2 - test3).apply(lambda x:x.days)
        test5 = test4 >= 30
        test5_nan = pd.Series(list(self.as_tf_data.user_id.value_counts()[self.as_tf_data.user_id.value_counts() == 1].index.get_level_values(0)))
        self.user_data['is_churn'] = self.user_data['risk_level'] == 4 
        self.user_data.is_churn = self.user_data['id'].isin(test5[test5 == True].index.get_level_values(0).unique())
        for i in test5_nan:
            self.user_data[self.user_data.id == i]['is_churn'] = True
        churn_rate_wk = []
        for i in range(len(self.user_id_by_parameter)):
#             test6 = test5[test5[test5 == True].index.get_level_values(0).isin(self.user_id_by_parameter[i][1])] # Didn't do transation for 30 days
            test6 = self.user_id_by_parameter[i][1].isin(test5[test5 == True].index.get_level_values(0))
            test7 = test5_nan.isin(self.user_id_by_parameter[i][1]).sum() # Only did one transaction

            churn_rate_wk.append((len(test6[test6==True].index.get_level_values(0).unique())+test7)/len(self.user_id_by_parameter[i][1]))

        return churn_rate_wk


    def plot_confusion_matrix(self,cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')


        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
