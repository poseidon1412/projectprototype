# Core packages
import streamlit as st 
# Images packages
from PIL import Image
# EDA packages
import pandas as pd 
import numpy as np
# Data visualisation packages
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
# Machine Leaning Models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

img = Image.open('mutemaEnterprises.jpg')
st.set_page_config(page_title='Datasets and Machine Learning Models Analyzer',page_icon = img)

def main():
	def modelsf():
		ml_models = ['Linear Regression','Logistic Regression','SVM','KNN','Random Forest']
		selected_model = st.sidebar.selectbox('Machine Learning Models',ml_models)
		if selected_model == 'Linear Regression':
			st.write('''# Linear Regression''')
			try:
				clf = LinearRegression()
				clf.fit(X_train,y_train)
				accuracy = clf.score(X_test,y_test)
				st.write('ACCURACY is {}'.format(accuracy))
		
			except Exception as e:
				st.error(
					'Selected algorithm does not support inserted dataset \n'
					'The data inserted is not compatible with the algorithm selected for example you have selected a regression algorithm '
					'but your dataset include strings which cannot be parsed to floats or integers thus since the regression algorithm works '
					'with numbers and not other datatypes, it will give an error please select another algorithm................. \n'
					'PLEASE SELECT ANOTHER ALGORITHM')

		elif selected_model == 'Logistic Regression':
			st.write('''# Logistic Regression''')
			try:
				clf = LogisticRegression(random_state=rndm_state)
				clf.fit(X_train,y_train)
				accuracy = clf.score(X_test,y_test)
				st.write('ACCURACY is {}'.format(accuracy))
		
			except Exception as e:
				st.error(
					'Selected algorithm does not support inserted dataset \n'
					'The data inserted is not compatible with the algorithm selected for example you have selected a regression algorithm '
					'but your dataset include strings which cannot be parsed to floats or integers thus since the regression algorithm works '
					'with numbers and not other datatypes it will give an error please select another algorithm.......................... \n'
					'PLEASE SELECT ANOTHER ALGORITHM.')
	
		elif selected_model == 'SVM':
			st.write('''# SVM''')
			try:
				C = st.sidebar.slider('C',0.01,10.0)
				clf = SVC(C=C)
				clf.fit(X_train,y_train)
				accuracy = clf.score(X_test,y_test)
				st.write('ACCURACY is {}'.format(accuracy))
		
			except Exception as e:
				st.error(
					'Selected algorithm does not support inserted dataset \n'
					'The data inserted is not compatible with the algorithm selected for example you have selected a regression algorithm '
					'but your dataset include strings which cannot be parsed to floats or integers thus since the regression algorithm works '
					'with numbers and not other datatypes it will give an error please select another algorithm.......................... \n'
					'PLEASE SELECT ANOTHER ALGORITHM.')
			
		elif selected_model == 'KNN':
			st.write('''# KNN''')
			try:
				K = st.sidebar.slider('K',1,15)
				clf = KNeighborsClassifier(n_neighbors=K)
				clf.fit(X_train,y_train)
				accuracy = clf.score(X_test,y_test)
				st.write('ACCURACY is {}'.format(accuracy))
		
			except Exception as e:
				st.error(
					'Selected algorithm does not support inserted dataset \n'
					'The data inserted is not compatible with the algorithm selected for example you have selected a regression algorithm '
					'but your dataset include strings which cannot be parsed to floats or integers thus since the regression algorithm works '
					'with numbers and not other datatypes it will give an error please select another algorithm.......................... \n'
					'PLEASE SELECT ANOTHER ALGORITHM.')	

		elif selected_model == 'Random Forest':
			st.write('''Random Forest''')
			try:
				max_depth = st.sidebar.slider('max-depth',2,15)
				n_estimators  = st.sidebar.slider('n_estimators',1,100)
				clf = RandomForestClassifier(n_estimators = n_estimators,max_depth = max_depth,random_state=rndm_state)
				clf.fit(X_train,y_train)
				accuracy = clf.score(X_test,y_test)
				st.write('ACCURACY is {}'.format(accuracy))
		
			except Exception as e:
				st.error(
					'Selected algorithm does not support inserted dataset \n'
					'The data inserted is not compatible with the algorithm selected for example you have selected a regression algorithm '
					'but your dataset include strings which cannot be parsed to floats or integers thus since the regression algorithm works '
					'with numbers and not other datatypes it will give an error please select another algorithm.......................... \n'
					'PLEASE SELECT ANOTHER ALGORITHM.')

	def explanatory_data_analysis():
		# Show Dataset
		if st.checkbox('Show Dataset'):
			number = st.number_input('Number of rows to view',1)
			st.dataframe(df.head(number))
		if st.checkbox('Show number of rows and columns'):
			st.write(f'Rows: {df.shape[0]}')
			st.write(f'Columns: {df.shape[1]}')
		# if st.checkbox('Value Count'):
		if st.checkbox('Show Value Counts of Target Columns'):
			st.write(df.iloc[:,-1].value_counts())
		# Show Columns
		if st.checkbox('Column Labels'):
			st.write(df.columns)
		# Show Data Types 
		if st.checkbox('Data Types'):
			st.write(df.dtypes)
		# Selected Columns
		if st.checkbox('Select multiple colums'):
			all_columns = df.columns.tolist()
			selected_columns = st.multiselect('Select Columns',all_columns)
			selected_columns_df = df[selected_columns]
			if len(selected_columns) > 0:
				st.dataframe(selected_columns_df)
		# Show Summary
		if st.checkbox('Summary'):
			st.write(df.describe().T)


	def visualisation():
		# Plot and Visualisation
		# Correlation
		# Seaborn Plot
		if st.checkbox('Show Correlation Matrix with Heatmap'):
			if st.button('Generate Correlation Matix'):
				st.write('### Heatmap')
				fig, ax = plt.subplots(figsize=(10,10))
				st.write(sns.heatmap(df.corr(), annot=True,linewidths=0.5))
				st.pyplot(fig)

		# Pie Chart
		if st.checkbox('Pie Chart Plot of Target Columns'):
			if st.button('Generate Pie Plot'):
				st.success('Generating a Pie Chart Plot')
				st.write(df.iloc[:,-1].value_counts().plot.pie(autopct="%1.1f%%"))
				st.pyplot()
		# Customizable Plot
		plot_types = ['area','bar','line','hist','box','kde']
		type_of_plot = st.selectbox('Select Type of Plot',plot_types)
		all_cols = df.columns.tolist()
		selected_cols = st.multiselect('Select Columns to Plot',all_cols)
		cust_data = df[selected_cols]
		if st.button('Generate Plot'):
			st.success('Generating Plot of {} for the selected columns which are {}'.format(type_of_plot,selected_cols))
			if type_of_plot == 'area':
				st.area_chart(cust_data)
			elif type_of_plot == 'bar':
				st.bar_chart(cust_data)
			elif type_of_plot == 'line':
				st.line_chart(cust_data)

			elif type_of_plot:
				cust_plot = cust_data.plot(kind = type_of_plot)
				st.write(cust_plot)
				st.pyplot()

		# Count Plot	  
		if st.checkbox('Value Counts Plot'):
			st.text('Value Counts by Target')
			all_column_name = df.columns.tolist()
			primary_col = st.selectbox('Primary Colums to groupby',all_column_name)	
			selected_column_name = st.multiselect('Select Column',all_column_name)
			if st.button('Generate Value Counts Plot'):
				st.success('Generating Plot')
				if selected_column_name:
					value_count_plot = df.groupby(primary_col)[selected_column_name].count()
				else:
					value_count_plot = df.iloc[:,-1].value_counts()
				st.write(value_count_plot.plot(kind = 'bar'))
				st.pyplot()


	st.set_option('deprecation.showPyplotGlobalUse', False)
	menu = ['Explanatory Data Analysis','Feature Engineering','Machine Learning Models','About']
	choice = st.sidebar.selectbox('Main Menu',menu)
	if choice == 'Explanatory Data Analysis':
		st.write('''# Explanatory Data Analysis''')
		file_formats_types = ['csv','xlsx','json','txt']
		datafile_format = st.selectbox('Dataset File Format',file_formats_types)
		data_file = st.file_uploader('Upload',type = [datafile_format])	
		if data_file is not None:
			if datafile_format == 'csv':
				df = pd.read_csv(data_file)
			elif datafile_format == 'xlsx':
				df = pd.read_excel(data_file)
			elif datafile_format == 'txt':
				df = pd.read_csv(data_file)
			elif datafile_format == 'json':
				df = pd.read_json(data_file)
			else:
				st.info('Invalid Format')
			explanatory_data_analysis()

	elif choice == 'Feature Engineering':
		st.write('''# Feature Engineering''')
		file_formats_types = ['csv','xlsx','json','txt']
		datafile_format = st.selectbox('Dataset File Format',file_formats_types)
		data_file = st.file_uploader('Upload',type = [datafile_format])	
		if data_file is not None:
			if datafile_format == 'csv':
				df = pd.read_csv(data_file)
			elif datafile_format == 'xlsx':
				df = pd.read_excel(data_file)
			elif datafile_format == 'txt':
				df = pd.read_csv(data_file)
			elif datafile_format == 'json':
				df = pd.read_json(data_file)
			else:
				st.info('Invalid Format')
			visualisation()

	elif choice == 'Machine Learning Models':
		st.write('''# Machine Learning Models''')
		file_formats_types = ['csv','xlsx','json','txt']
		datafile_format = st.selectbox('Dataset File Format',file_formats_types)
		data_file = st.file_uploader('Upload',type = [datafile_format])	
		if data_file is not None:
			if datafile_format == 'csv':
				df = pd.read_csv(data_file)
			elif datafile_format == 'xlsx':
				df = pd.read_excel(data_file)
			elif datafile_format == 'txt':
				df = pd.read_csv(data_file)
			elif datafile_format == 'json':
				df = pd.read_json(data_file)
			else:
				st.info('Invalid Format')
			X = df.iloc[:,0:-1]
			y = df.iloc[: , -1]
			tst_sz = st.sidebar.slider('Test Size',1,100)
			rndm_state = st.sidebar.slider('Random State')
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tst_sz, random_state=rndm_state)
			modelsf()

	elif choice == 'About':
		st.write('''# About''')
		st.write("""
		# HCT204-Artificial Intelligence CTHSC 2021 Level 2.2 Project

		This project was developed by Takudzwa Mutema a student at the University of Zimbabwe currently studying Computer
		Science Degree(CTHSC).This web application helps to automate explanatory data analysis, feature engineering and 
		machine learning model training returning the accuracy which may help in choosing a more suitable model for a given dataset
		thus a model that gives more accuracy.The web application was developed using Python.It allows for a user to insert a dataset
		of format that is accepted by the web app(csv,txt,json,xlsx).After uploading a dataset one can experient with it using different
		data analysis supported and Graph ploting. If a dataset is already prepared for training model one can use the web app to find
		which model is more appropriate by looking at the accuracy.

		# Below is the Question which the web application tried to answer

		Create a website using python. The site should provide a platform whereby
		users should be able to upload their data sets. State the data formats that you
		support on that site. Lead the user through a tutorial process whereby a user is
		supposed to do Exploratory data analysis and feature engineering. After the
		Exploratory Data analysis and feature engineering stage, the user is supposed
		to pick any one of the machine learning models that you support on your
		platform. If a user picks a wrong machine learning model the system should
		give an informative response. The user must also be given a choice to compare
		accuracy

		email-takudzwamutema5@gmail.com
		""")

if __name__ == '__main__':
	main()
