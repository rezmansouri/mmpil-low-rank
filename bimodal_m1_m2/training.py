import torch


def predict(modalities, y, loss_fn, model):
    with torch.no_grad():
        out = model(modalities)
        loss = loss_fn(out, y)
        return out, loss


def fit(train_loader, val_loader, model, loss_fn, optimizer, lr_sched, lr_max_strikes, n_epochs, model_save_path, device):
    train_losses, val_losses = [], []
    m_train, m_val = len(train_loader.dataset), len(val_loader.dataset)
    best_val_loss = torch.tensor(float('inf'))
    lr_strike = 0
    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0
        for batch in train_loader:
            modalities, y = batch
            out = model(modalities)
            loss = loss_fn(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        lr_strike += 1

        val_loss = 0
        model.eval()
        for batch in val_loader:
            modalities, y = batch
            _, val_batch_loss = predict(
                modalities, y, loss_fn, model)
            val_loss += val_batch_loss.item()

        train_loss /= m_train
        train_losses.append(train_loss)
        val_loss /= m_val
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            print('best model', val_loss)
            torch.save(model, f'{model_save_path}/{epoch}.pth')
            best_val_loss = val_loss
            lr_strike = 0

        if lr_strike > lr_max_strikes:
            lr_sched.step()
            print('reducing lr to', optimizer.param_groups[0]['lr'])
            lr_strike = 0

        print(
            f"epoch {epoch:>3}\ttraining loss: {train_loss:0.5f}\tval loss: {val_loss:0.5f}")

    return train_losses, val_losses, model
