## Machine Learning: Linear Regression Model

### Zillow Dataset Analysis
----

##### Mijail Q. Mariano
##### github.com/mijailmariano

<br>

### **<u>Project Description & Goals:</u>**

The purpose of this analysis is to explore a Zillow Real Estate dataset through visualizations and statistical tests to identify potential features that can accurately predict future tax assessed home values.

After exploring the Zillow dataset I design a machine learning (ML) model using regression techniques that predicts a home's tax assessed value with **~16% greater accuracy** than a baseline mean estimate.


----
#### **<u>Home Value Exploration Questions & Hypothesis:</u>**

**1. Does the home location matter?**

I presume location matters in determining the final value/price of a home, as there could be economic factors (e.g., cost-of-living/inflationary considerations) that may be considered when assessing a home's relative value. Additionally, homes typically closer to metropolitan or labor producing areas may be more costly than rural/less-labor promoted areas.

**2. Does the size of the home matter?**

I presume that the larger the home (measured by sq. feet), the higher the home value will be. 

A larger home can be more attractive to home buyers since the space can act as both an initial family home, but also their “forever home”. Meaning that people who might not initially have a use for the additional space - may see the potential benefits of having it when they are ready to either 1. expand their family/or find use for the space or 2. View the space as a future investment for a buyer willing to pay the same or more for their spacious home in the future (investment thinking).


**3. Does the home purchase period matter?**

I presume that the period when a home is purchased or placed on the market will matter. Home buyers may be more reluctant to purchase a home in the colder regional months (e.g., typically winter) when the weather may be less favorable for moving. 

There may also be *renter factors or periods in the year when leases end and renters make the decision to purchase a home, subsequently driving more buyers to the market and thus potentially increasing home sale values. 

----

### <u>**Statistically Significant Model Features**</u>

</br>

<u>**``Continuous Features/Variables:``**</u>

|feature|corr. coefficient| p-value|
|----|----|----|
|home_age|-0.2707|0.0|
|latitude|-0.1389|0.0|
|living_sq_feet|0.4589|0.0|

<br>

<u>**``Discrete/Categorical Features/Variables:``**</u>

|feature|t-score|p-value|
|----|----|----|
|1_to_3.5_baths|-5.8721|0.0|
|1_to_2_bedrooms|-25.5077|0.0|
|half_bathroom|24.2692|0.0|
|fips_code_6037.0*|-16.7577|0.0|
|fips_code_6059.0|20.5924|0.0|
|fips_code_6111.0|8.6164|0.0|
|purchase_month_April|6.6364|0.0|
|purchase_month_January|-4.3544|0.0|
|purchase_month_July|5.4294|0.0|
|purchase_month_May|-5.7304|0.0|
|purchase_month_September|4.1576|0.0|
|purchase_month_March|-2.5398|0.01|
|purchase_month_August|-2.6040|0.01|


*fips_code_6037 (LA, California) not selected in final model features* *

----

### **<u>Model Results</u>**

* Training R-squared w/Linear Regression: 0.2442
* Training R-squared w/Lasso Lars: 0.2442
* Training R-squared w/Tweedie Regressor: 0.1655

<br>

|Model|Train RMSE|Validate RMSE|
|----|----|----|
|linear regression|178296.33|178914.2|
|lasso lars|17298.37|178915.13|
|tweedie regressor|187359.38|187427.43|
|baseline mean|205092.78|204730.82|

<br>

<center>

<u>**``Linear Regression Deployment and Performance Through Test Dataset``**</u>



|Linear Model Performance|RMSE|Relative % Difference|
|----|----|----|
|Baseline|204730.70|0.00|
|Train|178296.33|0.15|
|Validate|178914.20|0.00|
|Test (final)|177111.14|0.01|

</center>

----

### **<u>``Analysis Summary``</u>**

Overall, the linear regression model performed at ~16% better accuracy than a baseline mean home value predictor. Though not entirely conclusive of a home's tax assessed value - I believe this model may be able to handle fluctuations in the overall market particularly well. 

By using "binned" or categorical features in traditionally sought after home characteristics (e.g., bedrooms, bathrooms) to determine a home's value, the model helps to handle external factors such as seasonality/seasonal effects, cultural preferences, or demand shocks that can undoubtedly impact the number of 'for sale' homes available and subsquently, house prices in relatively short periods.


### **``Recommendations:``**

1. Create a **Real-estate Training Program** that aims at helping Real-estate Brokers/Agents, Marketing, and Real-estate Consultancy teams to familiarize themselves with seasonal patterns in their local areas. By offering this program to real-estate professionals who are often closest to both sellers and buyers, it would help them to:

 - Better advise their clients on the most optimal periods to enter the market (purchase/sell their home)
 - More quickly recognize housing market shocks and make real-time decisions that help to normalize home value prices 

2. Use our online, mobile application, and advisory platforms to promote renovations of older homes, specifically smaller spaces where they may gain to benefit converting these spaces into **half-baths**. 

   - My analysis showed that homes with half-baths along with having **more finished living space** appear to be attractive characteristics for home buyers who may be trading in traditionally sought after home characteristis such as the number of bedrooms or bathrooms for more efficient and univarsal home space.

----

### **Looking Ahead (next steps):**

- Improve the model's predictive accuracy by identifying, testing, and including other potential home market factors 
    - regional cost-of-living indices
    - unemployment rates
    - educational/school ratings
    - crime rates
    - home design styles
<br></br>
- Calculate home/area distance to nearby metropolitan cities, park/recreational areas, schools, hospitals/hospice centers, etc.
  
- Parse out fips codes into more distinct locations either by cities/towns/villages/or even exact neighborhoods

----
### **<u>Repository Roadmap:</u>**

Below is a file breakdown on how to best navigate this GitHub repository and the subsequent analysis. All code, data synthesis, and models can be found here for future reproduction or improvements:

1. **final_report.ipynb**

   Data Science pipeline overview document. Project artifact of the hypothesis tests conducted, regression techniques used for machine learning models, key analysis takeaways, and recommendations.

2. **data_exploration.ipynb**

   Jupyter Notebook used as the data science pipeline file which walks-through the process for creating the necessary data acquisition and cleaning functions.

3. **acquire.py** 
  
   Python module file that imports the neccessary Zillow real-estate dataset and subsequent features outputs for modeling phase in the analysis. If just using or running the final_report.ipynb file, then this corresponding acquire.py file should suffice and the prepare.py file is not needed.
   
   **Note:** you must first import the initial Zillow dataset from MySQL. When passed in the "acquire.get_zillow_dataset()" function, the data is then stored locally as a .csv file - then referenced as a pd.Dataframe thereafter. 

4. **prepare.py**

     Python module file with created functions needed to clean, and manipulate the Zillow dataset used in this analysis.

<br>

----
### **<u>Initial Questions for Hypothesis Testing:</u>**

<br>

**1. Does the home location matter?**

I presume location matters in determining the final value/price of a home, as there could be economic factors (e.g., cost-of-living/inflationary considerations) that may be considered when assessing a home's relative value. Additionally, homes typically closer to metropolitan or labor producing areas may be more costly than rural/less-labor promoted areas.

* Homes closer to schools (predicted positive relationship)
* Home closer to parks/recreational areas (predicted positive relationship)
* Homes near hospitals/hospice care (predicted positive relationship)
* Areas with relatively less crime rates or high law-enforcement presence. (predicted negative relationship)
* Given the additional insurance costs/natural disaster considerations homes near bodies of water may be less attractive to home buyers. (predicted negative relationship)

``Relative Features:``
1. Fips
2. Latitude 
3. Longitude 
4. Parcel_id
5. Region_City_id
6. Region_id_County
7. Region_id_Zip

<br>

**2. Does the size of the home matter?**

I presume that the larger the home (as measured by sq. ft.), the higher the home value will be. 

A larger home can be more attractive to home buyers since the space can act as both an initial family home, but also their “forever home”. Meaning that people who might not initially have a use for the additional space - may see the potential benefits of having it when they are ready to either 1. expand their family/or find use for the space or 2. View the space as a future investment for a buyer willing to pay the same or more for their spacious home in the future (investment thinking). 

* Consider the total number of baths 
* Consider the total number of bedrooms

``Relative Features:``
1. Calculated_Finished_Sq_Feet
2. Bathroom_Count
3. Bedroom_Count
4. Full_Bath_Count

<br>

**3. Does the home purchase period matter?**

I presume that the period when a home is purchased or placed on the market will matter. Home buyers may be more reluctant to purchase a home in the colder regional months (e.g., typically winter) when the weather may be less favorable for moving. 

There may also be *renter factors or periods in the year when leases end and renters make the decision to purchase a home, subsequently driving more buyers to the market and thus potentially increasing home sale values. 

*(higher demand + “same” or not enough supply = more competition/higher home purchase price)*

* Consider seasonal patterns (e.g., summer months vs. winter) 

``Relative Features:``
1. Transaction_Date
2. Year_built
3. Assessment_Year

----
## <center> **Data Dictionary** </center>

home_age:
- total age of the home in years

latitude:
- relative north/south location of the home

living_sq_feet:
-  calculated total finished living area of the home 

1_to_3.5_baths:
- home has 1-3.5 bathrooms

1_to_2_bedrooms:
- home has 1-2 bedrooms

half_bathroom:
- home has min. of 1 half bathroom (contains sink and/or toilet only)

fips_code_6037:
- federal info. processing standard code (Los Angeles, California)

fips_code_6059:
- federal info. processing standard code (Orange, California)

fips_code_6111:
- federal info. processing standard code (Ventura, California)

county_id_1286:
- Humboldt County California

county_id_2061:
- Madera County California

county_id_3101:
- Placer County California

purchase_month_April:
- home transaction/purchased in the month of April

purchase_month_August:
- home transaction/purchased in the month of August

purchase_month_January:
- home transaction/purchased in the month of January

purchase_month_July:
- home transaction/purchased in the month of July

purchase_month_March:
- home transaction/purchased in the month of March

purchase_month_May:
- home transaction/purchased in the month of May

purchase_month_September:
- home transaction/purchased in the month of September



