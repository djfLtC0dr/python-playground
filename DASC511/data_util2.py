
class UnitOfMeasurement:
    def __init__(self,name,shorthand,scaler_to_base = 1, offset_to_base=0,other_acceptable_labels=None):
        self.name = name
        self.shorthand = shorthand
        self.scaler_to_base = scaler_to_base
        self.offset_to_base = offset_to_base
        self.other_acceptable_labels = other_acceptable_labels

    def convert_value_to_base(self, value):
        return self.scaler_to_base * value + self.offset_to_base

    def convert_value_from_base(self, value):
        return (1.0/self.scaler_to_base) * value - self.offset_to_base

    def format_value(self, value):
        return "{} {}".format(value, self.shorthand)


class UnitOfMeasurementProvider:

    def __init__(self, default_unit : UnitOfMeasurement):
        self.default_unit = default_unit
        self.unit_label_map = {}

    def get_base_unit(self) -> UnitOfMeasurement:
        return self.default_unit

    def lookup_unit_for_unit_label(self, unit_label):
        unit_of_measurement = self.unit_label_map[unit_label]
        
        return unit_of_measurement

    def register_unit(self,unit_of_measurement,make_default=False):
        if make_default:
            self.default_unit = unit_of_measurement
        
        self.unit_label_map[unit_of_measurement.name] = unit_of_measurement
        self.unit_label_map[unit_of_measurement.shorthand] = unit_of_measurement
        if unit_of_measurement.other_acceptable_labels is not None:
            for other_label in unit_of_measurement.other_acceptable_labels:
                self.unit_label_map[other_label] = unit_of_measurement
    
    def load_from_csv_file(self,unit_of_measurement_file_path):

        with open(unit_of_measurement_file_path) as csv_file:
            import csv
            reader = csv.reader(csv_file,delimiter=",")
            skip_header = False
            first = True
            for row in reader:
                if not skip_header:
                    skip_header = True
                else:
                    name = row[0]
                    shorthand = row[1]
                    scaler_to_base = float(row[2])
                    offset_to_base = float(row[3])
                    other_acceptable_label = row[4]

                    self.register_unit(UnitOfMeasurement(name,shorthand,scaler_to_base,offset_to_base,other_acceptable_labels=tuple([other_acceptable_label])),first)
                    first = False

class Measurement:
    def __init__(self,value,unit_of_measurement : UnitOfMeasurement):
        self.value = value
        self.unit_of_measurement = unit_of_measurement

    def __add__(self,rhs):
        base_unit = unit_of_measurement_provider.get_base_unit()
        lhs_value = self.unit_of_measurement.convert_value_to_base(self.value)
        rhs_value = rhs.unit_of_measurement.convert_value_to_base(rhs.value)
        return Measurement(lhs_value + rhs_value, base_unit)

    def __sub__(self,rhs):
        base_unit = unit_of_measurement_provider.get_base_unit()
        lhs_value = self.unit_of_measurement.convert_value_to_base(self.value)
        rhs_value = rhs.unit_of_measurement.convert_value_to_base(rhs.value)
        return Measurement(lhs_value - rhs_value, base_unit)

    def __mul__(self,rhs):
        return Measurement(self.value * rhs, self.unit_of_measurement)
    
    def __rmul__(self,lhs):
        return Measurement(self.value * lhs, self.unit_of_measurement)

    def __str__(self):
        return self.unit_of_measurement.format_value(self.value)

    def convert_units(self, new_unit_of_measurement : UnitOfMeasurement):
        base_value = self.unit_of_measurement.convert_value_to_base(self.value)
        self.value = new_unit_of_measurement.convert_value_from_base(base_value)
        self.unit_of_measurement = new_unit_of_measurement

    @staticmethod
    def Parse(value:str):
        parts = value.split(" ")
        n_value = float(parts[0])
        unit_label = parts[1]
        unit = unit_of_measurement_provider.lookup_unit_for_unit_label(unit_label)
        return Measurement(n_value,unit)


#meters_unit = UnitOfMeasurement("meters","m")
#feet_unit = UnitOfMeasurement("feet","ft",scaler_to_base=1/2.931)
#inches_unit = UnitOfMeasurement("inches","in",scaler_to_base=1/(2.931*12))

unit_of_measurement_provider = UnitOfMeasurementProvider(Measurement)

unit_of_measurement_provider.load_from_csv_file("DASC511/length_measurements.csv")

#unit_of_measurement_provider.register_unit(meters_unit,True)
#unit_of_measurement_provider.register_unit(feet_unit)
#unit_of_measurement_provider.register_unit(inches_unit)

measurement_1 = Measurement.Parse("3 ft")
# print("Measurement 1 {}".format(measurement_1))

measurement_2 = Measurement.Parse("1.5 m")
# print("Measurement 2 {}".format(measurement_2))

measurement_3 = (measurement_1 + measurement_2)
# print("Measurement 3 {}".format(measurement_3))

measurement_4 = measurement_1 * 3
# print("Measurement 4 {}".format(measurement_4))

measurement_5 = measurement_2 - measurement_1
# print("Measurement 5 {}".format(measurement_5))

meters_unit = unit_of_measurement_provider.lookup_unit_for_unit_label("m")
# print("Meters unit: {}".format(meters_unit.name))

measurement_6 = Measurement.Parse("3 ft")
measurement_6.convert_units(meters_unit)
# print("Measurement 6 {}".format(measurement_6))

inches_unit = unit_of_measurement_provider.lookup_unit_for_unit_label("inch")
# print("Inches unit: {}".format(inches_unit.name))

measurement_7 = Measurement.Parse("3 ft")
measurement_7.convert_units(inches_unit)
# print("Measurement 7 {}".format(measurement_7))

km_unit = unit_of_measurement_provider.lookup_unit_for_unit_label("km")
measurement_8 = Measurement.Parse("1500 m")
measurement_8.convert_units(km_unit)
# print("Measurement 8 {}".format(measurement_8))


