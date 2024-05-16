#!/bin/bash


models=(
    "AmazonCat-13K--0.7--0.1724481980349503.pickle"
    "AmazonCat-13K--0.71--0.17876073668249523.pickle"
    "AmazonCat-13K--0.72--0.18519788849705168.pickle"
    "AmazonCat-13K--0.73--0.1919101023532594.pickle"
    "AmazonCat-13K--0.74--0.19904129736887738.pickle"
    "AmazonCat-13K--0.75--0.2064911206128473.pickle"
    "AmazonCat-13K--0.76--0.2142559713220387.pickle"
    "AmazonCat-13K--0.77--0.22230999074063157.pickle"
    "AmazonCat-13K--0.78--0.2306249841703397.pickle"
    "AmazonCat-13K--0.79--0.23914905065626502.pickle"
    "AmazonCat-13K--0.8--0.2484625988375904.pickle"
    "AmazonCat-13K--0.81--0.2588980948305496.pickle"
    "AmazonCat-13K--0.82--0.2695696098777636.pickle"
    "AmazonCat-13K--0.83--0.2809636021589641.pickle"
    "AmazonCat-13K--0.84--0.2927757170829797.pickle"
    "AmazonCat-13K--0.85--0.3055143211958774.pickle"
    "AmazonCat-13K--0.86--0.3191886213103781.pickle"
    "AmazonCat-13K--0.87--0.3341034356043724.pickle"
    "AmazonCat-13K--0.88--0.350202011827318.pickle"
    "AmazonCat-13K--0.89--0.3681072481564285.pickle"
    "AmazonCat-13K--0.9--0.3884501942123101.pickle"
    "AmazonCat-13K--0.91--0.41103931477211425.pickle"
    "AmazonCat-13K--0.92--0.4361475677189066.pickle"
    "AmazonCat-13K--0.93--0.46416358793198925.pickle"
    "AmazonCat-13K--0.94--0.49728847793147496.pickle"
    "AmazonCat-13K--0.95--0.537049540236426.pickle"
    "AmazonCat-13K--0.96--0.589516877989551.pickle"
    "AmazonCat-13K--0.97--0.6597164219487569.pickle"
    "AmazonCat-13K--0.98--0.7293031348441131.pickle"
    "AmazonCat-13K--0.99--0.8725519027526369.pickle"
)

#!/bin/bash

# Specify the source directory
source_dir="/mnt/HDD-Seagate-16TB-2/justinchanchan8/tree-prunning-results/runs/1vsrest/AmazonCat-13K/0-100-quantile/models/"

# Specify the destination
destination="justinchanchan8@marcie.csie.ntu.edu.tw:/home/justinchanchan8/models/1vsrest/AmazonCat13K"


# Loop through the files and copy each one
for file in "${models[@]}"; do
    scp "$source_dir$file" "$destination"
done

