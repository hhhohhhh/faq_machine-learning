
# python运行脚本的三种模式 :
- 普通模式运行，不会自动去加载测试用例执行
- unittest 测试框架运行模式，可以自动去发现testcase并执行
- pytest 测试框架运行模式，就是我们上面2个步骤都是使用pytest测试框架运行的

## pycharm 如何修改脚本运行的模式呢:
file > setting > Tools > Python Integated Tools > Testing: Default 

注意：  
tests目录下 不能使用 test_*** 目录 ，但是可以使用一般的目录 ***

# pytest 简介
pytest是python的一种单元测试框架，与python自带的unittest测试框架类似，但是比unittest框架使用起来更简洁，效率更高。
并且pytest兼容unittest的用例，支持的插件也更多


# 用例设计原则:
    文件名以test_*.py文件和*_test.py
    以test_开头的函数
    以Test开头的类
    以test_开头的方法
    所有的包pakege必须要有__init__.py文件

# pytest运行规则：
查找当前目录及其子目录下以test_*.py或*_test.py文件，找到文件后，在文件中找到以test开头函数并执行。

# 执行用例
    1.执行某个目录下所有的用例: pytest 文件名/
    2.执行某一个py文件下用例: pytest 脚本名称.py
    3.-k 按关键字匹配 : pytest -k "MyClass and not method"

# pytest 命令行 
pytest [file_path] 指定文件  
  - -q 是静默模式，不会打印用例输出  
  - -v 打印用例执行的详细/简略过程  
  - -s 运行过程中执行print打印函数  