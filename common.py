"""
Client-Server Communications
"""
import asyncio

class Message:
    def __init__(self, polling_freqency=0.01):
        self.polling_freqency = polling_freqency
        self.queue = asyncio.Queue()

    """ 
    Give: Method takes three arguments and a shared workload queue. 
    Adds a mapping - a key value pair tag:value into the queue which
     can later be used by the other client.
    ```queue = [{tag: value}]```
    """
    async def give(self, tag, value, op_id='anonymous'):
        tag = op_id + ':' + tag.replace(':', '\\:')
        if self.queue.empty():
            tags = {tag: value}
            await self.queue.put(tags)
        else:
            tags = await self.queue.get()
            if tags.get(tag) is None:
                tags[tag] = value
            await self.queue.put(tags)

    """ 
    Get: Method takes a tag, a shared workload queue
     and optionally an ID for the parent operation. 
    Gets the tagged element or polls till it is populated and returns.
    """
    async def get(self, tag, op_id='anonymous'):
        tag = op_id + ':' + tag.replace(':', '\\:')
        q = self.queue._queue
        while self.queue.empty() or tag not in q[0]:
            await asyncio.sleep(self.polling_freqency)
        return q[0][tag]
