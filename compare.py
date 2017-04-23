from itertools import izip

line = 0
for sln, pln in izip(open('sequentialoutput.dat', 'r'), open('parlleloutput.dat', 'r')):
	line += 1
        if (sln is '\n' or pln is '\n' or sln.strip() is '' or pln.strip() is ''):
		continue

	snm = float(sln[:4])
	pnm = float(pln[:4])
        sexp = int(sln[-2:])
	pexp = int(pln[-2:])

	snm = snm * (10**sexp)
	pnm = pnm * (10**pexp)

	if (abs(snm-pnm) > 10000):
		print ('INVALID FOUND ON LINE: %s\ns: %sp: %s' % (line,sln,pln))
		#break



