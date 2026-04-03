using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;

namespace PlatformPractice;

public sealed class WorkItem
{
    public required string Id { get; init; }
    public required string TenantId { get; init; }
    public int Attempts { get; set; }
}

public sealed class ResilientWorker
{
    private readonly Random _random = new(42);
    private readonly Queue<WorkItem> _queue = new();
    private readonly List<WorkItem> _deadLetter = new();

    private const int MaxRetries = 3;

    public IReadOnlyCollection<WorkItem> DeadLetter => _deadLetter.AsReadOnly();

    public void Enqueue(WorkItem item) => _queue.Enqueue(item);

    public async Task RunAsync(CancellationToken ct)
    {
        while (_queue.Count > 0 && !ct.IsCancellationRequested)
        {
            var item = _queue.Dequeue();
            var success = await ProcessWithPolicyAsync(item, ct);

            if (!success)
            {
                if (item.Attempts > MaxRetries)
                {
                    _deadLetter.Add(item);
                }
                else
                {
                    _queue.Enqueue(item);
                }
            }
        }
    }

    private async Task<bool> ProcessWithPolicyAsync(WorkItem item, CancellationToken ct)
    {
        item.Attempts++;

        using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        timeoutCts.CancelAfter(TimeSpan.FromMilliseconds(300));

        var sw = Stopwatch.StartNew();
        try
        {
            var ok = await SimulateRemoteCallAsync(timeoutCts.Token);
            Console.WriteLine($"id={item.Id} tenant={item.TenantId} attempts={item.Attempts} latency_ms={sw.ElapsedMilliseconds} ok={ok}");
            return ok;
        }
        catch (OperationCanceledException)
        {
            Console.WriteLine($"id={item.Id} timeout attempts={item.Attempts}");
            return false;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"id={item.Id} err={ex.Message} attempts={item.Attempts}");
            return false;
        }
        finally
        {
            sw.Stop();
        }
    }

    private async Task<bool> SimulateRemoteCallAsync(CancellationToken ct)
    {
        var simulatedLatencyMs = _random.Next(40, 360);
        await Task.Delay(simulatedLatencyMs, ct);

        // ~20% transient failure
        return _random.NextDouble() > 0.20;
    }
}

public static class Demo
{
    public static async Task Main()
    {
        var worker = new ResilientWorker();

        for (var i = 0; i < 40; i++)
        {
            worker.Enqueue(new WorkItem
            {
                Id = $"task-{i:000}",
                TenantId = $"tenant-{i % 4}",
            });
        }

        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        await worker.RunAsync(cts.Token);

        Console.WriteLine($"dead_letter_count={worker.DeadLetter.Count}");
    }
}
