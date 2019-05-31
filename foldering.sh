while read name;
 	do
 	dir_name=`echo "$name" | cut -d "_" -f 3` 
	#each_day=`echo "$name" |awk -F '_' '{print $1"_"$2"_"$3}'`
	echo  "one_subject/""$name"
	echo $dir_name
	mv "one_subject/""$name" $dir_name"/"
done < names
