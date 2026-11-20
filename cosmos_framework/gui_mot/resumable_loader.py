"""Resume the deterministic GUI stream at the trainer's consumed-fetch count."""

from collections import deque

from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader


class GuiPackingDataLoader(PackingDataLoader):
    def set_start_iteration(self, iteration):
        if self.max_samples_per_batch != 1:
            raise ValueError("GUI cursor resume requires exactly one sample per batch")
        inner = self.dataloader_list[0]
        if inner.dataloader.num_workers != 0:
            raise ValueError("GUI cursor resume requires num_workers=0")
        inner.dataset.set_start_sample(iteration)
        # Discard prewarmed samples and iterators created before checkpoint load.
        self.dataloaders = [iter(inner)]
        self.buffers = [deque()]
        super().set_start_iteration(iteration)
