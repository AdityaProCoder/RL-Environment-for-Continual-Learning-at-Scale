import http.client, ssl, urllib3, requests, sys
urllib3.disable_warnings()

_orig_https_init = http.client.HTTPSConnection.__init__
def _new_https_init(self, *args, **kwargs):
    kwargs['context'] = ssl._create_unverified_context()
    return _orig_https_init(self, *args, **kwargs)
http.client.HTTPSConnection.__init__ = _new_https_init

_old_send = requests.Session.send
def _new_send(self, request, **kwargs):
    kwargs['verify'] = False
    return _old_send(self, request, **kwargs)
requests.Session.send = _new_send

from kagglepipe.cli import main

if __name__ == "__main__":
    sys.argv = ["kagglepipe"] + sys.argv[1:]
    main()
