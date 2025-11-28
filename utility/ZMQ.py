import zmq
import json
import sys 

class ZMQCommunicator:
    """ZMQ通信封装类, 负责处理与外部的通信"""
    
    # 1. 修改默认地址为 TCP，并重命名参数为更通用的 'address'
    def __init__(self, address="tcp://127.0.0.1:5555"):
        """初始化ZMQ通信
        
        Args:
            address: ZMQ绑定的地址 (例如: "tcp://127.0.0.1:5555" 或 "ipc:///tmp/...")
        """
        self.address = address
        self.context = None
        self.socket = None
        self._initialize()
    
    def _initialize(self):
        """初始化ZMQ上下文和socket"""
        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PUB)
            # 使用 self.address 进行绑定
            self.socket.bind(self.address)
            print(f"ZMQ发布者启动, 绑定到:{self.address}")
        except zmq.ZMQError as e:
            print(f"ZMQ初始化失败: {e}")
            self.cleanup()
            # 2. 初始化失败时直接退出程序，防止后续错误
            print("程序因ZMQ初始化失败而终止。")
            sys.exit(1)
    
    # 3. 修正类型提示，因为我们发送的是列表(list)
    def send_data(self, data: list):
        """发送数据
        
        Args:
            data: 要发送的数据（例如关节角度列表）
        """
        if not self.socket:
            print("ZMQ socket未初始化, 无法发送数据")
            return
            
        try:
            # 转换为JSON字符串并发布
            json_data = json.dumps(data)
            self.socket.send_string(json_data)
        except zmq.ZMQError as e:
            print(f"发送数据失败: {e}")
    
    def cleanup(self):
        """清理ZMQ资源"""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        print("ZMQ资源已释放")