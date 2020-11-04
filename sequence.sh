cat $1 | while read line
do
    echo $line
    name="../data/sequences/"$line"_imgs"
    name2="../data/sequences/"$line"_pcaed_sequenced"
    if [ -f "$name2" ];
    then
        echo $name2" already exist"
        continue
    else
        echo "Staterd generating sequence for donor: "$line
        grep "sara_img/"$line $2 > $name
        python3 decom_sequence_generator_keras_pcaed.py --path $name --donor_id $line 
    fi
done
