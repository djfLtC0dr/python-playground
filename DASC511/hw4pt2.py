from data_util2 import Measurement, UnitOfMeasurement, UnitOfMeasurementProvider

#(1 point) Assign a variable named name a string object with text representing your name.
name: str = "Dan Fawcett"
print("Coded by: {}".format(name))

#(2 points) Assign a variable measurement_1 the returned object of the Measurement's Parse static method when passed “730.3 ft”
measurement_1 = Measurement.Parse("730.3 ft")
print("Measurement 1 {}".format(measurement_1))

#(2 points) Assign a variable measurement_2 the returned object of the Measurement's Parse static method when passed “1.5 km”
measurement_2 = Measurement.Parse("1.5 km")
print("Measurement 2 {}".format(measurement_2))

#(2 points) Assign a variable measurement_3 the result of adding measurement_1 and measurement_2 together using the ‘+’ operator
measurement_3 = measurement_1 + measurement_2
print("Measurement 3 {}".format(measurement_3))

#(2 points) Assign a variable measurement_4 the result of multiplying measurement_1 by 3 using the multiplication, ‘*’, operator
measurement_4 = measurement_1 * 3
print("Measurement 4 {}".format(measurement_4))

#(2 points) Assign a variable measurement_5 the result of subtracting measurement_1 from measurement_2 using the subtraction, ‘-’, operator
measurement_5 = measurement_2 - measurement_1
print("Measurement 5 {}".format(measurement_5))

#(2 points) Use the unit_of_measurement_provider variable within the data_util2 module to lookup the UnitOfMeasurement object representing meters. 
# Assign the returned UnitOfMeasurement object to a variable meters_unit
unit_of_measurement_provider = UnitOfMeasurementProvider(Measurement)
unit_of_measurement_provider.load_from_csv_file("DASC511/length_measurements.csv")
#meters_unit = UnitOfMeasurement("meters","m")
#unit_of_measurement_provider.register_unit(meters_unit)
meters_unit = unit_of_measurement_provider.lookup_unit_for_unit_label("meters")
print("Meters unit: {}".format(meters_unit.name))

#(2 points) Assign a variable measurement_6 the returned object of the Measurement's Parse static method when passed “53.4 ft”. 
# Then call the convert_units method on this Measurement object passing in the UnitOfMeasurement object representing meters, meters_unit.
measurement_6 = Measurement.Parse("53.4 ft")
measurement_6.convert_units(meters_unit)
print("Measurement 6 {}".format(measurement_6))

#(2 points) Use the unit_of_measurement_provider variable within the data_util2 module to lookup the UnitOfMeasurement object representing inches. 
# Assign the returned UnitOfMeasurement  object to a variable inches_unit
#inches_unit = UnitOfMeasurement("inches","in")
#unit_of_measurement_provider.register_unit(inches_unit)
inches_unit = unit_of_measurement_provider.lookup_unit_for_unit_label("inches")
print("Inches unit: {}".format(inches_unit.name))

#(2 points) Assign a variable measurement_7 the returned object of the Measurement's Parse static method when passed “42.0 ft”. 
# Then call the convert_units method on this Measurement object passing in the UnitOfMeasurement object representing inches, inches_unit.
measurement_7 = Measurement.Parse("42.0 ft")
measurement_7.convert_units(inches_unit)
print("Measurement 7 {}".format(measurement_7))