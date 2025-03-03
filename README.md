# Air Compressor predictive maintenance using ML

Notebook link: https://colab.research.google.com/drive/1Q2n_c4ODnUbItGiBd712_5gjqitWOSpN#scrollTo=NfpkY_QoQQK-

In this project, we explore the use of machine learning to predict and maintain the performance of air compressor systems. Air compressors are essential devices that convert power into potential energy stored in compressed air, which can then be utilized for various applications such as powering pneumatic tools, inflating tires, and operating machinery. Ensuring air compressors’ optimal performance and longevity is crucial for many industries, and predictive maintenance can help identify potential failures and improve the maintenance process.

## Application type
We will predict the bearings status in the air compressor system, a binary variable (0 or 1). Therefore, this is a classification project.

The goal here is to model the bearings’ status based on the features of the air compressor system for its subsequent use in predictive maintenance.

## Data set
The data file air_compressor_maintenance.csv contains the information for the air compressor example. This dataset consists of measurements taken from a compressor system supplying air to a factory production line, with 17 features collected in total. The dataset comprises 17 variables (columns) and 1000 instances (rows).

The features or variables included in the dataset are as follows:

* **RPM**: Indicates the number of rotations per minute for the motor.
* **Motor Power**: Measures the power consumption of the electric motor in kilowatts.
* **Torque**: Provides the torque produced by the motor in Newton-meter.
* **Outlet Pressure Bar**: Denotes the outlet pressure of compressed air in bars.
* **Air Flow**: Displays the flow rate of compressed air in cubic meters per minute.
* **Noise dB**: Represents the noise level of the compressor system in decibels.
* **Outlet Temp**: Shows the outlet temperature of the compressed air in degrees Celsius.
* **Water Pump Outlet Pressure**: Gives the outlet pressure of the water pump in bars.
* **Water Inlet Temp**: Specifies the inlet temperature of cooling water in degrees Celsius.
* **Water Outlet Temp**: Provides the outlet temperature of cooling water in degrees Celsius.
* **Water Pump Power**: Measures the power consumption of the water pump in kilowatts.
* **Water Flow**: Indicates the cooling water flow rate in cubic meters per minute.
* **Oil Pump Power**: This represents the power consumption of the oil pump in kilowatts.
* **Oil Tank Temp**: Shows the temperature of the oil tank in degrees Celsius.
* **Ground Acceleration**: Represents the acceleration experienced by the compressor at its mounting point, measured in the X, Y, and Z directions in meters per second squared.
* **Head Acceleration**: Refers to the acceleration value measured at the compressor head bolt or upper cooling fin in the X, Y, and Z directions, typically expressed in gravitational units.
* **Bearings Status**: Indicates the condition of the bearings in the motor and compressor system. The values can be ‘Ok’ for properly functioning bearings or ‘Noise’ for bearings that may need maintenance or replacement due to wear or damage affecting the performance and efficiency of the compressor.
