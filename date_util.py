import datetime
from datetime import date
import csv

# Problem 1 (1 point)

# Enter your name below
name = "Dan Fawcett"

# Problem 2 (5 points)

# Implement a function named "parse_date_string" that takes one parameter named
# "date_str" of type "str". Sometimes this parameter only specifies a year or
# a year and a month without specifing a day. See test problems below. A dash
# character, '-', seperates the date components. If the text has 0 dash
# characters then only a year is provided and a month and day should be filled
# in as the first month and first day of year. If the text has 1 dash character
# then only a year and month are provided and the day should be filled in as the
# first day. The code should create a datetime.date object from this text string.
# See https://docs.python.org/3/library/datetime.html#datetime.date for details
# about how to create this object. The created datetime.date object should be returned
# to the function caller. The month text and day text may be two digits padded a 0 if a
# single digit number.

def split_str(date_str):
    return(date_str.split('-'))

#TODO implement
def parse_date_string(date_str):
  ret_date = date(2022, 4, 13)
  if date_str.count('-') == 0:
    # first month and first day of year
    ret_date = date(int(date_str), 1, 1)
  elif date_str.count('-') == 1:
    yr_mo = split_str(date_str)
    #day should be filled in as the first day.
    ret_date = date(int(yr_mo[0]), int(yr_mo[1]), 1)
  elif date_str.count('-') == 2:
    yr_mo_dd = split_str(date_str)
    ret_date = date(int(yr_mo_dd[0]), int(yr_mo_dd[1]), int(yr_mo_dd[2]))
  return ret_date


d1 = parse_date_string("2002")
assert isinstance(d1,datetime.date)
assert d1.year == 2002
assert d1.month == 1
assert d1.day == 1


d2 = parse_date_string("2021-03")
assert isinstance(d2,datetime.date)
assert d2.year == 2021
assert d2.month == 3
assert d2.day == 1

d3 = parse_date_string("2020-3-7")
assert isinstance(d3,datetime.date)
assert d3.year == 2020
assert d3.month == 3
assert d3.day == 7

d4 = parse_date_string("2019-12-25")
assert isinstance(d4,datetime.date)
assert d4.year == 2019
assert d4.month == 12
assert d4.day == 25


# Problem 3 (5 points)

# Implement a function named "read_dataset_csv_file" that takes one parameter
# named "file_path" of type "str" and returns a "list" object containing "tuple"
# objects for each line in the csv file. The file_path parameter should reference
# a CSV file containing the dataset text. The first column in CSV file is a number,
# the second column is a number, and the third column is date text. For each line, 
# convert the first two columns values to numbers and the thrid column values to date
# objects with the function above.
# See the provided test.txt file for an example testing file.

def recursively_apply(l, f):
  lst_tuples = []  
  for i in l:
    if len(i) == 1:
        lst_tuples.append(int(i))
    else:
        lst_tuples.append(f(i))
  return lst_tuples

#TODO implement
def read_dataset_csv_file(file_path):
  with open(file_path) as csvfile:
    csvReader = csv.reader(csvfile, delimiter=',')
    lst_tuples = []
    for row in csvReader:
      lst_tuples.append(recursively_apply(row, parse_date_string))
    return lst_tuples

# To test problem 3 you can uncomment the following. This does not test
# the list having the correct values, only the proper object types.
# You should have the test.txt in the same working directory

test_dataset = read_dataset_csv_file("test.txt")
assert len(test_dataset) == 4

#assert isinstance(test_dataset[0],tuple)
#assert isinstance(test_dataset[1],tuple)
#assert isinstance(test_dataset[2],tuple)
#assert isinstance(test_dataset[3],tuple)

#assert isinstance(test_dataset[0][0],(float,int))
#assert isinstance(test_dataset[1][0],(float,int))
#assert isinstance(test_dataset[2][0],(float,int))
#assert isinstance(test_dataset[3][0],(float,int))

#assert isinstance(test_dataset[0][1],(float,int))
#assert isinstance(test_dataset[1][1],(float,int))
#assert isinstance(test_dataset[2][1],(float,int))
#assert isinstance(test_dataset[3][1],(float,int))

#assert isinstance(test_dataset[0][2],datetime.date)
#assert isinstance(test_dataset[1][2],datetime.date)
#assert isinstance(test_dataset[2][2],datetime.date)
#assert isinstance(test_dataset[3][2],datetime.date)