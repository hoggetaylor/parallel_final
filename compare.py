from itertools import izip

line = 0
for sln, pln in izip(open('outseq.dat', 'r'), open('outpar.dat', 'r')):
        if (sln is '\n' or pln is '\n'):
		continue
	line += 1

	snm = float(sln[:4])
	pnm = float(pln[:4])
        sexp = int(sln[-2:])
	pexp = int(pln[-2:])

	snm = snm * (10**sexp)
	pnm = pnm * (10**pexp)

	if (abs(snm-pnm) > 10000):
		print ('INVALID FOUND ON LINE: %s\n%s%s%f' % (line,sln,pln,abs(snm-pnm)))
		break



