Watauga River Level Predictor
=======================
![net](https://advguides.wpenginepowered.com/wp-content/uploads/Watauga-Gorge-Stateline-Falls-1536x864.png "taug")
 
 Introduction
 ------------
 
The Watauga River is one of the top summertime spots in Boone, flowing by town east to west into Tennessee. It forms from a spring near the base of Peak Mountain at Linville Gap in Avery County, and eventually empties its waters into the Gulf of Mexico. All sections of the Watauga in North Carolina are free flowing, with the first impedance to the river’s flow being the Tennessee Valley Authority dam at Watauga Lake just across the border. The most popular sections of the river are sections IV and V, situated about 30 minutes east of Boone. These sections hold class II to class V whitewater rapids, swimming holes, and spectacular fishing spots. Since the river is undammed above Watauga Lake, the river level in these sections varies greatly with the weather conditions. For everyone who enjoys the beautiful rivers close to town, the amount of water has a large impact on the opportunity for and quality of recreation. Without rain, the Watauga usually holds flow at around 100 cubic feet per second. This would be a great level for fishing or swimming. For whitewater kayaking however, the level needs to be at least 200 cubic feet per second, with the optimum range being 400-800. Fishermen, swimmers, kayakers, and anyone else looking to get out and enjoy the river all need to know the water level to guide their activities.
While there is a gauge that tells the current flow rate of the river on an hourly basis, it doesn’t have any type of forecasting ability to give a heads up when the river may run. The only way anyone can predict what the river level is going to do in the future is through skill reading the forecast and weather radar. This is a pretty effective solution if you know what to look for, but it can be difficult to do accurately considering the size of the river’s watershed. This project concerns the development of a time series forecasting machine-learning algorithm on the historical gauge and rainfall data in order to correlate local precipitation information with the height of the river. It takes into consideration the daily rainfall amounts at 3 different locations around the river’s watershed and uses them to estimate the water level. Machine learning seemed to be an obvious choice for this type of problem, as it has a very powerful ability to identify patterns across very large datasets. Using this model along with the weather forecast information, the algorithm is able to predict the river level for up to several weeks in advance. The model doesn’t strive to be to-the-hour accurate, but rather provides an average level for the entire day. With this type of forecast, river-goers can make better decisions about which days to plan on heading out. In order to make the estimated levels as easy to access as possible, the predictions for the next 10 days and the rainfall predictions used to make them are publicly hosted online at https://wataugapredictor.pythonanywhere.com. I have not seen any type of forecasting applied to the level of rivers, at least not in a short-term sense as would be used by your everyday outdoorsman. Having this information aids immensely in the planning of adventures on and around the river, primarily for kayakers, fisherman, and swimmers.
 
Features
--------

- __Generates predictions of river level 10 days in advance__
- __Provides rain forecast information used for predictions__
- __Level forecast generated from neural network back end__
- __Updates daily at 8:00am Eastern Time__

Description of Equipment
------------------------
- All that is needed is an internet-accessible device!
 
Using the Program
-----------------

*__Open your desired web browser on your device__*
    
*__Navigate to the following webpage:__*
    
    https://wataugapredictor.pythonanywhere.com
    
Contribution
----------
 
- __Issue Tracker:__ https://github.com/nobo428/TaugPredictor/issues
- __Source Code:__ https://github.com/nobo428/TaugPredictor
 
Support / Troubleshooting
-------
 
If you are having issues, please let me know!
*Email me at:* nbbodin4@gmail.com
