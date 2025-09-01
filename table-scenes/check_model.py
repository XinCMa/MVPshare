# 检查 YOLOE 模型的方法和属性
from ultralytics import YOLOE
import inspect

# 加载模型
model_path = "yoloe-11l-seg.pt"
model = YOLOE(model_path)

# 检查可用的方法和属性
print("===== YOLOE 模型对象的方法和属性 =====")
model_attrs = [attr for attr in dir(model) if not attr.startswith('_')]
print(f"方法和属性: {model_attrs}")

# 检查模型配置
print("\n===== 模型配置 =====")
if hasattr(model, 'overrides'):
    print(f"覆盖配置: {model.overrides}")

print("\n===== 可用的类别 =====")
# 检查模型的类别信息
if hasattr(model, 'names'):
    print(f"类别名称: {model.names}")
elif hasattr(model.model, 'names'):
    print(f"模型类别名称: {model.model.names}")

# 检查模型的类型
print("\n===== 模型类型 =====")
print(f"模型类型: {type(model)}")
print(f"model.model 类型: {type(model.model)}")

# 打印predict方法的签名
print("\n===== predict 方法签名 =====")
predict_signature = inspect.signature(model.predict)
print(f"predict签名: {predict_signature}")
