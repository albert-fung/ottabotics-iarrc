import time
from subprocess import call,check_output

number_of_runs = 5
count = 0

for i in range(number_of_runs):

	print "Starting..."

	start_time = time.time()
	check_output("python classify_image.py --image_file bottleongrass.jpg",shell=True)
	end_time = time.time()

	print "Elapsed time: " + str(end_time - start_time)
	count += (end_time - start_time)


print "Average: " + str(count / number_of_runs)
