#bash remove.sh to_be_removed_paths originalFile > lines_to_remove
#to_be_removed_paths only has the paths
#lines_to_remove has the lines with those paths. these lines would have the same format as the lines in originalFile
cat $1 | while read line
do
  grep "$line" $2 >> 2berem
done  

##Then do 
grep -vxFf 2berem $2 > $2'_remains'
rm 2berem
