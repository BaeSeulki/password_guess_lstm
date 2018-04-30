from functools import wraps
import time


# 使用装饰器来衡量函数执行时间
def fn_timer(functions):
    @wraps(functions)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = functions(*args, **kwargs)
        t1 = time.time()
        print("Total time running %s: %s seconds" % (functions.__name__, str(t1 - t0)))
        return result

    return function_timer
