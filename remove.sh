#bash remove.sh to_be_removed_paths originalFile > lines_to_remove
#to_be_removed_paths only has the paths
#lines_to_remove has the lines with those paths. these lines would have the same format as the lines in originalFile
cat $1 | while read line
do
  #echo $line
  grep "$line" $2
done  

##Then do 
#grep -vxFf lines_to_remove originalFile > originalFile_with_non_of_the_lines_with_tha_paths_to_be_removed
