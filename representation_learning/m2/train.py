import torch
import numpy as np


def predict(batch, enc, dec, loss_fn, device):
    with torch.no_grad():
        batch = batch.to(device)
        out = enc(batch)
        out = dec(out)
        loss = loss_fn(out, batch)
        return out, loss


def fit(train_loader, val_loader, enc, dec, loss_fn, optimizer, lr_sched, lr_max_strikes, n_epochs, model_save_path, device):
    train_losses, val_losses = [], []
    best_val_loss = torch.tensor(float('inf'))
    lr_strike = 0
    for epoch in range(1, n_epochs + 1):
        enc.train()
        dec.train()
        train_loss = []
        for batch in train_loader:
            batch = batch.to(device)
            out = enc(batch)
            out = dec(out)
            loss = loss_fn(out, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        lr_strike += 1

        val_loss = []
        enc.eval()
        dec.eval()
        for batch in val_loader:
            _, val_batch_loss = predict(
                batch, enc, dec, loss_fn, device)
            val_loss.append(val_batch_loss.item())

        train_loss = np.mean(train_loss)
        train_losses.append(train_loss)
        val_loss = np.mean(val_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            print('best model', val_loss)
            torch.save(enc.state_dict(), f'{model_save_path}/{epoch}_enc.pth')
            torch.save(dec.state_dict(), f'{model_save_path}/{epoch}_dec.pth')
            best_val_loss = val_loss
            lr_strike = 0

        if lr_strike > lr_max_strikes:
            lr_sched.step()
            print('reducing lr to', optimizer.param_groups[0]['lr'])
            lr_strike = 0

        print(
            f"epoch {epoch:>3}\ttraining loss: {train_loss:0.5f}\tval loss: {val_loss:0.5f}")

    return train_losses, val_losses, (enc, dec)
