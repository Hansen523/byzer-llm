from ..conversable_agent import ConversableAgent
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from ....utils.client import ByzerLLM
from byzerllm.utils.retrieval import ByzerRetrieval
from ..agent import Agent
from ray.util.client.common import ClientActorHandle, ClientObjectRef
import time
import ray
from .. import get_agent_name,run_agent_func,ChatResponse,modify_message_metadata,modify_message_content
from byzerllm.utils.client import default_chat_wrapper,LLMResponse
from byzerllm.apps.agent import Agents
from byzerllm.apps.agent.extensions.preview_file_agent import PreviewFileAgent
from byzerllm.apps.agent.extensions.python_codesandbox_agent import PythonSandboxAgent
from byzerllm.apps.agent.extensions.visualization_agent import VisualizationAgent
from byzerllm.apps.agent.assistant_agent import AssistantAgent
from byzerllm.apps.agent.common_agent import CommonAgent
from byzerllm.apps.agent.extensions.spark_sql_agent import SparkSQLAgent
from byzerllm.apps.agent.extensions.rhetorical_agent import RhetoricalAgent
from byzerllm.apps.agent.extensions.sql_reviewer_agent import SQLReviewerAgent

class DataAnalysisPipeline(ConversableAgent):  
    DEFAULT_SYSTEM_MESSAGE = '''You are a helpful data analysis assistant.
You don't need to write code, or anwser the question. The only thing you need to do 
is plan the data analysis pipeline.

You have following agents to use:

1. visualization_agent, 这个 Agent 可以帮助你对数据进行可视化。
2. assistant_agent, 这个 Agent 可以帮你生成代码对数据进行分析，统计。
3. common_agent, 这个Agent 只会根据对话来帮助用户分析数据。他不会生成任何代码去分析数据。
4. spark_sql_agent, 这个Agent 可以根据用户对话帮助用户作分析，它主要生成 Spark SQL 代码对数据进行分析。


Please check the user's question and decide which agent you need to use. And then reply the agent name only.
If there is no tool can help you, 
you should reply exactly `UPDATE CONTEXT`.
''' 

    DEFAULT_USER_MESSAGE = """
"""

    def __init__(
        self,
        name: str,        
        llm: ByzerLLM,
        retrieval: ByzerRetrieval, 
        file_path:str,
        file_ref:ClientObjectRef ,        
        chat_name:str,
        owner:str,
        system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,        
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Optional[str] = "NEVER",
        code_execution_config: Optional[Union[Dict, bool]] = False,
        **kwargs,
    ):
        super().__init__(
            name,
            llm,retrieval,
            system_message,
            is_termination_msg,
            max_consecutive_auto_reply,
            human_input_mode,
            code_execution_config=code_execution_config,            
            **kwargs,
        )
        self.chat_name = chat_name
        self.owner = owner
        self.file_path = file_path
        self.file_ref = file_ref        
        self._reply_func_list = []
        # self.register_reply([Agent, ClientActorHandle,str], ConversableAgent.generate_llm_reply)   
        self.register_reply([Agent, ClientActorHandle,str], DataAnalysisPipeline.run_pipeline) 
        self.register_reply([Agent, ClientActorHandle,str], DataAnalysisPipeline.reply_reheorical_agent) 
        self.register_reply([Agent, ClientActorHandle,str], ConversableAgent.check_termination_and_human_reply) 

        params = {}
        if "chat_wrapper" in kwargs:
            params["chat_wrapper"] = kwargs["chat_wrapper"]        

        self.python_interpreter = Agents.create_local_agent(PythonSandboxAgent,"python_interpreter",
                                                llm,retrieval,
                                                chat_name=self.chat_name,
                                                owner=self.owner,
                                                max_consecutive_auto_reply=100,
                                                system_message="you are a code sandbox",**params
                                                )
        

        self.preview_file_agent = Agents.create_local_agent(PreviewFileAgent,"privew_file_agent",llm,retrieval,
                                                                chat_name=self.chat_name,
                                                                owner=self.owner,
                                                                max_consecutive_auto_reply=100,
                                                                code_agent = self.python_interpreter,**params
                                        )
        
        self.visualization_agent = Agents.create_local_agent(VisualizationAgent,"visualization_agent",llm,retrieval,
                                                            chat_name=self.chat_name,
                                                            owner=self.owner,
                                                            max_consecutive_auto_reply=100,                                        
                                                            code_agent = self.python_interpreter,**params
                                        ) 
        self.assistant_agent = Agents.create_local_agent(AssistantAgent,"assistant_agent",llm,retrieval,
                                                        chat_name=self.chat_name,
                                                        owner=self.owner,
                                                        max_consecutive_auto_reply=100,
                                                        code_agent = self.python_interpreter,**params)  
        self.common_agent = Agents.create_local_agent(CommonAgent,"common_agent",llm,retrieval,
                                                       chat_name=self.chat_name,
                                                        owner=self.owner,
                                                        max_consecutive_auto_reply=100,
                                                        code_agent = self.python_interpreter,**params) 
        
        self.sql_reviewer_agent = Agents.create_local_agent(SQLReviewerAgent,"sql_reviewer_agent",llm,retrieval,chat_name=self.chat_name,
                                                        owner=self.owner,max_consecutive_auto_reply=100,**params
                                                            )
        
        self.spark_sql_agent = Agents.create_local_agent(SparkSQLAgent,"spark_sql_agent",llm,retrieval,
                                                         sql_reviewer_agent=self.sql_reviewer_agent,
                                                        chat_name=self.chat_name,
                                                        owner=self.owner,                                                        
                                                        max_consecutive_auto_reply=100,**params)   
        self.rhetoorical_agent = Agents.create_local_agent(RhetoricalAgent,"rhetoorical_agent",llm,retrieval,
                                                            chat_name=self.chat_name,
                                                            owner=self.owner,
                                                            max_consecutive_auto_reply=100,**params)                
        
        self.agents = {
            "assistant_agent":self.assistant_agent,
            "visualization_agent":self.visualization_agent,
            "common_agent":self.common_agent,
            "privew_file_agent":self.preview_file_agent,
            "python_interpreter":self.python_interpreter,
            "spark_sql_agent":self.spark_sql_agent,            
        } 
        self.reply_from_agent = {}       

    def get_agent_chat_messages(self,agent_name:str):
        return self.agents[agent_name].get_chat_messages()
    
    def update_system_message_by_agent(self, agent_name:str,system_message: str):
        if agent_name in self.agents:
            self.agents[agent_name].update_system_message(system_message)
            return True
        return False    

    def get_agent_names(self):
        return list(self.agents.keys())  

    def get_agent_system_message(self,agent_name:str):
        return self.agents[agent_name].system_message      

    def preview_file(self):
        self.preview_file_agent._prepare_chat(self.python_interpreter, True)        
        self.max_consecutive_auto_reply = 0          
        self.initiate_chat(
        self.preview_file_agent,
        message={
            "content":f"We have a file, the file path is: {self.file_path} , please preview this file",
            "role":"user",
            "metadata":{
                "file_path":self.file_path,
                "file_ref":self.file_ref
            }
        })      
        # sync the conversation of preview_file_agent to other agents
        print("sync the conversation of preview_file_agent to other agents",flush=True)
        for agent in self.agents.values():            
            for message in self._messages["privew_file_agent"]:                 
                self.send(message=message,recipient=agent,request_reply=False)

    def select_agent(self,raw_message,messages,sender):
        _,llm_reply = self.generate_llm_reply(raw_message,messages,sender)
        fail = "UPDATE CONTEXT" in llm_reply[-20:].upper() or "UPDATE CONTEXT" in llm_reply[:20].upper()
        if fail:
            return True, None
        else:
            return True,llm_reply.strip()
        
    def reply_reheorical_agent(
        self,
        raw_message: Optional[Union[Dict,str,ChatResponse]] = None,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Union[ClientActorHandle,Agent,str]] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, Dict, None,ChatResponse]]:
        if get_agent_name(sender) == "rhetoorical_agent":
            if messages is None:
                messages = self._messages[get_agent_name(sender)]
            m = messages[-1]
            metadata = m.get("metadata",{})
            if metadata.get("ask_user",False):
                self.reply_from_agent["rhetoorical_agent"] = m["content"]
                
            return True,None
        return False,None

    def run_pipeline(
        self,
        raw_message: Optional[Union[Dict,str,ChatResponse]] = None,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Union[ClientActorHandle,Agent,str]] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, Dict, None,ChatResponse]]:
                
        if messages is None:
            messages = self._messages[get_agent_name(sender)]
        
        ori_message = messages[-1]  
          
        self.send(message=ori_message,recipient=self.rhetoorical_agent)
        if self.reply_from_agent.get("rhetoorical_agent",None) is not None:
            v = self.reply_from_agent["rhetoorical_agent"]
            self.reply_from_agent["rhetoorical_agent"] = None
            return True, v

        _,agent_name = self.select_agent(raw_message,messages,sender)
        print(f"Select agent: {agent_name} to answer the question: {ori_message['content'][0:20]}",flush=True)
        
        if agent_name and agent_name in self.agents:
            agent = self.agents[agent_name]
            # reset the agent except the conversation history  
            self._prepare_chat(agent, clear_history=False)                      
            self.send(message=ori_message,recipient=agent,request_reply=False)                                                
            agent_reply = agent.generate_reply(raw_message=None,messages=None,sender=self)
            if isinstance(agent_reply,dict):
                agent_reply = agent_reply["content"]
            return True, {"content":agent_reply,"metadata":{"TERMINATE":True}}
        
        return self.generate_llm_reply(raw_message,messages,sender)

class DataAnalysisPipelineManager:
    def __init__(self) -> None:
        self.pipelines = {}
        self.lasted_updated = {}
    
    ## if the sandbox is not used for 1h, we will remove it
    def check_pipeline_timeout(self,timeout:int=60*60): 
        remove_names = []
        for name in self.lasted_updated:
            if time.time() - self.lasted_updated[name] > timeout:
                remove_names.append(name)
        for name in remove_names:
            del self.pipelines[name]
            del self.lasted_updated[name]        

    def check_pipeline_exists(self,name:str)->bool:
        self.check_pipeline_timeout() 
        return name in self.pipelines

    def get_pipeline(self,name:str):                
        self.check_pipeline_timeout()        
        return self.pipelines[name]
    
    def force_clear(self):
        self.pipelines = {}
        self.lasted_updated = {}

    def remove_pipeline(self,name:str):
        if name in self.pipelines:
            del self.pipelines[name]
            del self.lasted_updated[name]    

    def get_or_create_pipeline( self,
                                name:str,
                                llm:ByzerLLM,retrieval:ByzerRetrieval,
                                file_path:str,
                                file_ref:ClientObjectRef,
                                chat_name:str,
                                owner:str,
                                chat_wrapper:Optional[Callable[[ByzerLLM,Optional[List[Dict]],Dict],List[LLMResponse]]] = None,
                                num_gpus:int=0,num_cpus:int=0):
        self.lasted_updated[name] = time.time()
        self.check_pipeline_timeout()
        if name in self.pipelines:            
            return self.pipelines[name]
        
        pipeline = ray.remote(DataAnalysisPipeline).options(
                name=name,                                              
                num_cpus=num_cpus,
                num_gpus=num_gpus
            ).remote(
                name = name,
                llm = llm,
                retrieval = retrieval,
                file_path=file_path,
                file_ref=file_ref,
                chat_wrapper=chat_wrapper,
                chat_name=chat_name,
                owner=owner                
                )
        self.pipelines[name] = pipeline
        return pipeline
    