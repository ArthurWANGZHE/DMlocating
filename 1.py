import graphviz

# 创建Graph对象
graphviz_path = 'C:/Program Files/Graphviz/bin/'  # 替换为您的Graphviz可执行文件所在路径
dot = graphviz.Graph(format='png')

# 添加节点
dot.node('电源供应模块', shape='box')
dot.node('漏电保护器模块', shape='box')
dot.node('控制单元模块', shape='box')
dot.node('显示报警装置模块', shape='box')

# 添加边
dot.edge('电源供应模块', '漏电保护器模块')
dot.edge('漏电保护器模块', '控制单元模块')
dot.edge('控制单元模块', '显示报警装置模块')

# 设置图形属性
dot.attr(rankdir='LR')  # 设置布局方向为从左到右

# 保存并渲染图形
dot.render('整体设计框图', view=True)
