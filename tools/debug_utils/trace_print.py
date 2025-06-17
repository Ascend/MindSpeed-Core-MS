# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import sys
import traceback
import re
from contextlib import ContextDecorator


def print_list(extracted_list, file=None):
    """
    Print the list of stack frames as returned by extract_tb() or extract_stack(),
    formatted as a stack trace to the given file.
    """
    if file is None:
        file = sys.stderr
    for item in traceback.StackSummary.from_list(extracted_list).format():
        print(item, file=file, end="", flush=True)


def print_stack(f=None, limit=None, file=None):
    """Print a stack trace from its invocation point.

    The optional 'f' argument specifies an alternate stack frame at which to start.
    The optional 'limit' and 'file' arguments behave as in print_exception().
    """
    if f is None:
        # Skip the frame for print_stack itself
        f = sys._getframe().f_back
    print_list(traceback.extract_stack(f, limit=limit), file=file)


def show_stack():
    """
    Print the full current call stack (including this function itself)
    at any point in the code.
    """
    traceback.print_stack()


class _FuncCallTracer:
    """
    A tracer class:
      - If both literal_names and regex_list are empty: no tracing is done.
      - Otherwise: only when the function name is in literal_names
        or matches any pattern in regex_list will the stack be printed.
    """
    def __init__(self, literal_names, regex_list):
        # Set of exact names to match
        self.literal_names = set(literal_names)
        # Precompile the regular expressions
        self.regexes = [re.compile(p) for p in (regex_list or [])]

    def __call__(self, frame, event, arg):
        if event == 'call':
            name = frame.f_code.co_name
            # First do an O(1) check for literal name matches
            if name in self.literal_names:
                self._print(name, matched_by='literal')
            else:
                # Then perform regex matching on the small list of patterns
                for rx in self.regexes:
                    if rx.fullmatch(name):
                        self._print(name, matched_by=f'regex ({rx.pattern})')
                        break
        return self

    def _print(self, name, matched_by):
        # 1. Extract the full current call stack as a list of FrameSummary
        stack = traceback.extract_stack()
        # 2. Remove the last two frames (this _print call and extract_stack call)
        trimmed = stack[:-2]
        # 3. Print the trimmed stack
        print(f"\n--- Stack trace for call to {name!r} (matched by {matched_by}) ---", flush=True)
        print_list(trimmed)
        print(f"--- End of trace for {name!r} ---\n", flush=True)


class TraceContext(ContextDecorator):
    """
    Context manager / decorator:

    1. Default mode (no arguments):
       - __enter__ prints the full stack once
       - __exit__ prints the full stack once again

    2. Literal-name mode (pass positional args):
       with TraceContext('foo', 'bar'):
         # Only when foo() or bar() is called will the stack be printed

    3. Regex mode (pass regex=...):
       with TraceContext(regex=['foo.*', 'get_\\d+']):
         # Only when the function name fully matches any regex will the stack be printed

    4. Mixed mode:
       with TraceContext('init', regex=['foo.*']):
         # Calls named 'init' or matching 'foo.*' will both trigger printing

    Usage examples:
      with TraceContext():
          ...

      with TraceContext('foo', 'bar'):
          ...

      with TraceContext(regex=['foo.*']):
          ...

      with TraceContext('x', regex=['get_.*']):
          ...
    """
    def __init__(self, *literal_names, regex=None):
        self.literal_names = literal_names
        self.regex_list = regex or []
        # Determine whether we are in default (no tracing) mode
        self._default = (not literal_names and not regex)
        self._tracer = None

    def __enter__(self):
        if self._default:
            # Default mode: print full stack on enter
            print("\n--- TraceContext enter (default) ---")
            traceback.print_stack()
            print("--- End of enter trace ---\n")
        else:
            # Literal, regex, or mixed mode: start tracing
            self._tracer = _FuncCallTracer(self.literal_names, self.regex_list)
            sys.settrace(self._tracer)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._default:
            # Default mode: print full stack on exit
            print("\n--- TraceContext exit (default) ---")
            traceback.print_stack()
            print("--- End of exit trace ---\n")
        else:
            # Stop tracing
            sys.settrace(None)
        # Do not suppress any exceptions
        return False
