from termcolor import colored
from PIL import Image

def raise_exception(e, err_str):
	pkg_err_str = 'Exception raised in package pyAIUtil: {}'.format(err_str)                  
	print colored(pkg_err_str,'red')
	print colored('Error Number {}: {}'.format(e.errno, e.strerror),'red') 


def imread(filename):
	try:
		im = Image.open(filename)
		Image.load()
	
	except IOError as e:
		err_str = 'imread could not open file {}'.format(filename)
		raise_exception(e, err_str)
	
	return im
	