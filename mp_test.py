import time
import threading

def compute(x):
    # Do some computation with x
    y = x * x

# Record start time
start_time = time.time()

# Threaded version
# Create a thread and pass it the compute function and an argument
i = 0
while i < 100:
    t = threading.Thread(target=compute, args=(10,))
    print(t)

    # Start the thread
    t.start()

    # Wait for the thread to complete
    t.join()
    i+=1

# Record end time
end_time = time.time()

# Print elapsed time
elapsed_time = end_time - start_time
print("Threaded elapsed time: {} seconds".format(elapsed_time))

# Non-threaded version
# Record start time
start_time = time.time()


i = 0
while i < 100:
    compute(10)
    i+=1
# Record end time
end_time = time.time()

# Print elapsed time
elapsed_time = end_time - start_time
print("Non-threaded elapsed time: {} seconds".format(elapsed_time))
