#!/bin/bash
while read zz; do	
    oldId=$(echo "$zz" | cut -d'/' -f10)
    gt_label=$(echo "$zz" | cut -d':' -f2)
    startYear=$(echo "$zz" | cut -d'/' -f9 | grep -o '..$')
    newId=$(cat new_mappings.txt | grep $oldId | cut -d ':' -f 2)

    #path=$(echo "$zz" |cut -d' ' -f1)    
    #path=$path" Photos/"
    #echo $path
    #ls "$path" | grep -v "icon" | grep ".JPG" | while read p; do
    utid=$(echo "$zz" | sed 's/_.*//') 
    month=$(echo "$zz" | cut -d'_' -f2)
    day=$(echo "$zz" | cut -d'_' -f3)
    
    if ! ( echo "$month" | grep "0" > /dev/null )
    then
        [[ $month -lt 10 ]] && month="0$month"
    fi
    
    if ! ( echo "$day" | grep "0" > /dev/null )
    then
        [[ $day -lt 10 ]] && day="0$day"
    fi
	
	if echo "$zz"  | grep '(' > /dev/null 
	then
		if echo "$zz" | grep " " > /dev/null
		then 
			year=$(echo "$zz" |  sed 's/.*_//' | cut -d ' ' -f1 | grep -o '..$')
		else
			year=$(echo "$zz" | sed 's/.*_//' | cut -d '(' -f1 | grep -o '..$')
		fi
		number=$(echo "$zz" | sed 's/.*(//; s/).*//; s/\..*//')
	else 
		year=$(echo "$zz" | sed 's/.*_//;s/\.JPG//' | grep -o '..$')
		number="0"	
	fi

    if [[ "$number" -lt 10 ]]
    then
        number="0$number"
    fi

    if [ $year -gt $startYear ]
        then
      y=$((year - startYear))
        else
      y=0
        fi

    echo $newId'/'$newId$y$month$day.$number.JPG': '$gt_label


done < data/GTs-ut100-102-106-108-109-110-111
