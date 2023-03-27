for zbin in 2
	do
	for bincent in 0
		do
		python fit_nocov.py $zbin $bincent L61_150
		python fit_nocov.py $zbin $bincent L43_150
		python fit_nocov.py $zbin $bincent L61_090
		python fit_nocov.py $zbin $bincent L43_090
		done
	done

git add --all .
git commit -m "added plots"
git push origin main
